use std::fmt::Debug;

use crate::ans::{AnsState, Spec};
use crate::bit_reader::BitReader;
use crate::constants::{Bitlen, DeltaLookback, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{bins, Bin, DeltaEncoding, DynLatents};
use crate::{ans, bit_reader, bits, delta, read_write_uint};
use aligned::{Aligned, A32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Default here is meaningless and should only be used to fill in empty
// vectors.
#[derive(Clone, Copy, Debug)]
pub struct BinDecompressionInfo<L: Latent> {
  pub lower: L,
  pub offset_bits: Bitlen,
}

impl<L: Latent> BinDecompressionInfo<L> {
  fn new(bin: &Bin<L>) -> Self {
    Self {
      lower: bin.lower,
      offset_bits: bin.offset_bits,
    }
  }
}

#[derive(Clone, Debug)]
struct State<L: Latent> {
  // scratch needs no backup
  // TODO: use an arena and heap-allocate these?
  offset_bits_csum_scratch: Aligned<A32, [Bitlen; FULL_BATCH_N]>,
  offset_bits_scratch: Aligned<A32, [Bitlen; FULL_BATCH_N]>,
  lowers_scratch: Aligned<A32, [L; FULL_BATCH_N]>,
  ans_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_state: Vec<L>,
  delta_state_pos: usize,
}

impl<L: Latent> State<L> {
  #[inline]
  fn set_scratch(&mut self, i: usize, offset_bit_idx: Bitlen, info: &BinDecompressionInfo<L>) {
    unsafe {
      *self.offset_bits_csum_scratch.get_unchecked_mut(i) = offset_bit_idx;
      *self.offset_bits_scratch.get_unchecked_mut(i) = info.offset_bits;
      *self.lowers_scratch.get_unchecked_mut(i) = info.lower;
    };
  }
}

// LatentBatchDecompressor does the main work of decoding bytes into Latents
#[derive(Clone, Debug)]
pub struct LatentPageDecompressor<L: Latent> {
  // known information about this latent variable
  u64s_per_offset: usize,
  infos: Vec<BinDecompressionInfo<L>>,
  needs_ans: bool,
  decoder: ans::Decoder,
  delta_encoding: DeltaEncoding,
  pub maybe_constant_value: Option<L>,

  // mutable state
  state: State<L>,
}

impl<L: Latent> LatentPageDecompressor<L> {
  pub fn new(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: DeltaEncoding,
    ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
    stored_delta_state: Vec<L>,
  ) -> PcoResult<Self> {
    let u64s_per_offset = read_write_uint::calc_max_u64s(bins::max_offset_bits(bins));
    let infos = bins
      .iter()
      .map(BinDecompressionInfo::new)
      .collect::<Vec<_>>();
    let weights = bins::weights(bins);
    let ans_spec = Spec::from_weights(ans_size_log, weights)?;
    let decoder = ans::Decoder::new(&ans_spec);

    let (working_delta_state, delta_state_pos) = match delta_encoding {
      DeltaEncoding::None | DeltaEncoding::Consecutive(_) => (stored_delta_state, 0),
      DeltaEncoding::Lookback(config) => {
        delta::new_lookback_window_buffer_and_pos(config, &stored_delta_state)
      }
    };

    let mut state = State {
      offset_bits_csum_scratch: Aligned::<A32, _>([0; FULL_BATCH_N]),
      offset_bits_scratch: Aligned::<A32, _>([0; FULL_BATCH_N]),
      lowers_scratch: Aligned::<A32, _>([L::ZERO; FULL_BATCH_N]),
      ans_state_idxs: ans_final_state_idxs,
      delta_state: working_delta_state,
      delta_state_pos,
    };
    let needs_ans = bins.len() != 1;
    if !needs_ans {
      // we optimize performance by setting state once and never again
      let bin = &bins[0];
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        state.offset_bits_scratch[i] = bin.offset_bits;
        state.offset_bits_csum_scratch[i] = csum;
        state.lowers_scratch[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    let maybe_constant_value =
      if bins::are_trivial(bins) && matches!(delta_encoding, DeltaEncoding::None) {
        bins.first().map(|bin| bin.lower)
      } else {
        None
      };

    Ok(Self {
      u64s_per_offset,
      infos,
      needs_ans,
      decoder,
      delta_encoding,
      maybe_constant_value,
      state,
    })
  }

  // This implementation handles only a full batch, but is faster.
  #[cfg(target_arch = "x86_64")]
  #[inline(never)]
  unsafe fn decompress_full_ans_symbols(&mut self, reader: &mut BitReader) {
    // At each iteration, this loads a single u64 and has all ANS decoders
    // read a single symbol from it.
    // Therefore it requires that ANS_INTERLEAVING * MAX_BITS_PER_ANS <= 57.
    // Additionally, we're unpacking all ANS states using the fact that
    // ANS_INTERLEAVING == 4.
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let mut state_idxs = _mm_loadu_si128(self.state.ans_state_idxs.as_ptr() as *const __m128i);
    // let [mut state_idx_0, mut state_idx_1, mut state_idx_2, mut state_idx_3] =
    //   self.state.ans_state_idxs;
    let infos = self.infos.as_slice();
    let ans_nodes = self.decoder.nodes.as_slice();
    for base_i in (0..FULL_BATCH_N).step_by(ANS_INTERLEAVING) {
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      let packed_vec = _mm256_set1_epi64x(u64::cast_signed(packed));
      // I hate that I have to do this with a macro, but it gives a serious
      // performance gain. If I use a [AnsState; 4] for the state_idxs instead
      // of separate identifiers, it tries to repeatedly load and write to
      // the array instead of keeping the states in registers.
      // TODO: can't we exploit the interleaving to use SIMD?
      // macro_rules! handle_single_symbol {
      //   ($j: expr, $state_idx: ident) => {
      //     let i = base_i + $j;
      //     let node = unsafe { ans_nodes.get_unchecked($state_idx as usize) };
      //     let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << node.bits_to_read) - 1);
      //     let info = unsafe { infos.get_unchecked(node.symbol as usize) };
      //     self.state.set_scratch(i, offset_bit_idx, info);
      //     bits_past_byte += node.bits_to_read;
      //     offset_bit_idx += info.offset_bits;
      //     $state_idx = node.next_state_idx_base + ans_val;
      //   };
      // }
      // handle_single_symbol!(0, state_idx_0);
      // handle_single_symbol!(1, state_idx_1);
      // handle_single_symbol!(2, state_idx_2);
      // handle_single_symbol!(3, state_idx_3);

      let i0 = base_i + 0;
      let i1 = base_i + 1;
      let i2 = base_i + 2;
      let i3 = base_i + 3;

      let node0 = unsafe { ans_nodes.get_unchecked(_mm_extract_epi32(state_idxs, 0) as usize) };
      let node1 = unsafe { ans_nodes.get_unchecked(_mm_extract_epi32(state_idxs, 1) as usize) };
      let node2 = unsafe { ans_nodes.get_unchecked(_mm_extract_epi32(state_idxs, 2) as usize) };
      let node3 = unsafe { ans_nodes.get_unchecked(_mm_extract_epi32(state_idxs, 3) as usize) };

      let info0 = unsafe { infos.get_unchecked(node0.symbol as usize) };
      let info1 = unsafe { infos.get_unchecked(node1.symbol as usize) };
      let info2 = unsafe { infos.get_unchecked(node2.symbol as usize) };
      let info3 = unsafe { infos.get_unchecked(node3.symbol as usize) };

      let bits_past_byte0 = bits_past_byte + node0.bits_to_read;
      let bits_past_byte1 = bits_past_byte0 + node1.bits_to_read;
      let bits_past_byte2 = bits_past_byte1 + node2.bits_to_read;
      let bits_past_byte3 = bits_past_byte2 + node3.bits_to_read;

      let bits_past_byte_vec = _mm256_set_epi64x(
        bits_past_byte2 as i64,
        bits_past_byte1 as i64,
        bits_past_byte0 as i64,
        bits_past_byte as i64,
      );

      let bits_to_read_vec = _mm_set_epi32(
        u32::cast_signed(node3.bits_to_read),
        u32::cast_signed(node2.bits_to_read),
        u32::cast_signed(node1.bits_to_read),
        u32::cast_signed(node0.bits_to_read),
      );

      let offset_bit_idx0 = offset_bit_idx + info0.offset_bits;
      let offset_bit_idx1 = offset_bit_idx0 + info1.offset_bits;
      let offset_bit_idx2 = offset_bit_idx1 + info2.offset_bits;
      let offset_bit_idx3 = offset_bit_idx2 + info3.offset_bits;

      // Current complex approach
      let ans_val_left_vec = _mm256_srlv_epi64(packed_vec, bits_past_byte_vec);
      // let ans_val_left_vec_mask = _mm256_set1_epi64x(0xFFFF_FFFF);
      // let truncated = _mm256_and_si256(ans_val_left_vec, ans_val_left_vec_mask);
      // let low64 = _mm256_castsi256_si128(truncated);
      // let high128 = _mm256_extracti128_si256(truncated, 1); 
      // const SHUFFLE_MASK: i32 = 0b01000100;
      // let lo32 = _mm_shuffle_epi32(low64, SHUFFLE_MASK);
      // let hi32 = _mm_shuffle_epi32(high128, SHUFFLE_MASK);
      // let ans_val_left_vec = _mm_unpacklo_epi32(lo32, hi32);
      
      let ans_val_left_vec = _mm_set_epi32(
        _mm256_extract_epi64(ans_val_left_vec, 3) as i32,
        _mm256_extract_epi64(ans_val_left_vec, 2) as i32,
        _mm256_extract_epi64(ans_val_left_vec, 1) as i32,
        _mm256_extract_epi64(ans_val_left_vec, 0) as i32,
      );

      let ans_val_right_vec = _mm_sub_epi32(
        _mm_sllv_epi32(_mm_set1_epi32(1), bits_to_read_vec),
        _mm_set1_epi32(1),
      );

      let ans_val_vec = _mm_and_si128(ans_val_left_vec, ans_val_right_vec);

      // let ans_val0 = (packed >> bits_past_byte) as AnsState & ((1 << node0.bits_to_read) - 1);
      // let ans_val1 = (packed >> bits_past_byte0) as AnsState & ((1 << node1.bits_to_read) - 1);
      // let ans_val2 = (packed >> bits_past_byte1) as AnsState & ((1 << node2.bits_to_read) - 1);
      // let ans_val3 = (packed >> bits_past_byte2) as AnsState & ((1 << node3.bits_to_read) - 1);

      *self.state.offset_bits_csum_scratch.get_unchecked_mut(i0) = offset_bit_idx;
      *self.state.offset_bits_csum_scratch.get_unchecked_mut(i1) = offset_bit_idx0;
      *self.state.offset_bits_csum_scratch.get_unchecked_mut(i2) = offset_bit_idx1;
      *self.state.offset_bits_csum_scratch.get_unchecked_mut(i3) = offset_bit_idx2;
      *self.state.offset_bits_scratch.get_unchecked_mut(i0) = info0.offset_bits;
      *self.state.offset_bits_scratch.get_unchecked_mut(i1) = info1.offset_bits;
      *self.state.offset_bits_scratch.get_unchecked_mut(i2) = info2.offset_bits;
      *self.state.offset_bits_scratch.get_unchecked_mut(i3) = info3.offset_bits;
      *self.state.lowers_scratch.get_unchecked_mut(i0) = info0.lower;
      *self.state.lowers_scratch.get_unchecked_mut(i1) = info1.lower;
      *self.state.lowers_scratch.get_unchecked_mut(i2) = info2.lower;
      *self.state.lowers_scratch.get_unchecked_mut(i3) = info3.lower;

      state_idxs = _mm_add_epi32(
        _mm_set_epi32(
          node3.next_state_idx_base as i32,
          node2.next_state_idx_base as i32,
          node1.next_state_idx_base as i32,
          node0.next_state_idx_base as i32,
        ),
        ans_val_vec,
        // _mm_set_epi32(
        //   ans_val3 as i32,
        //   ans_val2 as i32,
        //   ans_val1 as i32,
        //   ans_val0 as i32,
        // )
      );
      // state_idx_0 = node0.next_state_idx_base + ans_val0;
      // state_idx_1 = node1.next_state_idx_base + ans_val1;
      // state_idx_2 = node2.next_state_idx_base + ans_val2;
      // state_idx_3 = node3.next_state_idx_base + ans_val3;

      bits_past_byte = bits_past_byte3;
      offset_bit_idx = offset_bit_idx3;
    }

    reader.stale_byte_idx = stale_byte_idx;
    reader.bits_past_byte = bits_past_byte;
    // self.state.ans_state_idxs = [state_idx_0, state_idx_1, state_idx_2, state_idx_3];
    _mm_storeu_si128(
      self.state.ans_state_idxs.as_mut_ptr() as *mut __m128i,
      state_idxs,
    );
  }

  // This implementation handles arbitrary batch size and looks simpler, but is
  // slower, so we only use it at the end of the page.
  #[inline(never)]
  unsafe fn decompress_ans_symbols(&mut self, reader: &mut BitReader, batch_n: usize) {
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let mut state_idxs = self.state.ans_state_idxs;
    for i in 0..batch_n {
      let j = i % ANS_INTERLEAVING;
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      let node = unsafe { self.decoder.nodes.get_unchecked(state_idxs[j] as usize) };
      let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << node.bits_to_read) - 1);
      let info = &self.infos[node.symbol as usize];
      self.state.set_scratch(i, offset_bit_idx, info);
      bits_past_byte += node.bits_to_read;
      offset_bit_idx += info.offset_bits;
      state_idxs[j] = node.next_state_idx_base + ans_val;
    }

    reader.stale_byte_idx = stale_byte_idx;
    reader.bits_past_byte = bits_past_byte;
    self.state.ans_state_idxs = state_idxs;
  }

  #[inline(never)]
  unsafe fn decompress_offsets<const MAX_U64S: usize>(
    &mut self,
    reader: &mut BitReader,
    dst: &mut [L],
  ) {
    let base_bit_idx = reader.bit_idx();
    let src = reader.src;
    let state = &mut self.state;
    for (dst, (&offset_bits, &offset_bits_csum)) in dst.iter_mut().zip(
      state
        .offset_bits_scratch
        .iter()
        .zip(state.offset_bits_csum_scratch.iter()),
    ) {
      let bit_idx = base_bit_idx + offset_bits_csum as usize;
      let byte_idx = bit_idx / 8;
      let bits_past_byte = bit_idx as Bitlen % 8;
      *dst = bit_reader::read_uint_at::<L, MAX_U64S>(src, byte_idx, bits_past_byte, offset_bits);
    }
    let final_bit_idx = base_bit_idx
      + state.offset_bits_csum_scratch[dst.len() - 1] as usize
      + state.offset_bits_scratch[dst.len() - 1] as usize;
    reader.stale_byte_idx = final_bit_idx / 8;
    reader.bits_past_byte = final_bit_idx as Bitlen % 8;
  }

  #[inline(never)]
  fn add_lowers(&self, dst: &mut [L]) {
    for (&lower, dst) in self.state.lowers_scratch[0..dst.len()]
      .iter()
      .zip(dst.iter_mut())
    {
      *dst = dst.wrapping_add(lower);
    }
  }

  // If hits a corruption, it returns an error and leaves reader and self unchanged.
  // May contaminate dst.
  pub unsafe fn decompress_batch_pre_delta(&mut self, reader: &mut BitReader, dst: &mut [L]) {
    if dst.is_empty() {
      return;
    }

    if self.needs_ans {
      let batch_n = dst.len();
      assert!(batch_n <= FULL_BATCH_N);

      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader);
      } else {
        self.decompress_ans_symbols(reader, batch_n);
      }
    }

    // this assertion saves some unnecessary specializations in the compiled assembly
    assert!(self.u64s_per_offset <= read_write_uint::calc_max_u64s(L::BITS));
    match self.u64s_per_offset {
      0 => {
        dst.copy_from_slice(&self.state.lowers_scratch[..dst.len()]);
        return;
      }
      1 => self.decompress_offsets::<1>(reader, dst),
      2 => self.decompress_offsets::<2>(reader, dst),
      3 => self.decompress_offsets::<3>(reader, dst),
      _ => panic!(
        "[LatentBatchDecompressor] data type too large (extra u64's {} > 2)",
        self.u64s_per_offset
      ),
    }

    self.add_lowers(dst);
  }

  pub unsafe fn decompress_batch(
    &mut self,
    delta_latents: Option<&DynLatents>,
    n_remaining_in_page: usize,
    reader: &mut BitReader,
    dst: &mut [L],
  ) -> PcoResult<()> {
    let n_remaining_pre_delta =
      n_remaining_in_page.saturating_sub(self.delta_encoding.n_latents_per_state());
    let pre_delta_len = if dst.len() <= n_remaining_pre_delta {
      dst.len()
    } else {
      // If we're at the end, this won't initialize the last
      // few elements before delta decoding them, so we do that manually here to
      // satisfy MIRI. This step isn't really necessary.
      dst[n_remaining_pre_delta..].fill(L::default());
      n_remaining_pre_delta
    };
    self.decompress_batch_pre_delta(reader, &mut dst[..pre_delta_len]);

    match self.delta_encoding {
      DeltaEncoding::None => Ok(()),
      DeltaEncoding::Consecutive(_) => {
        delta::decode_consecutive_in_place(&mut self.state.delta_state, dst);
        Ok(())
      }
      DeltaEncoding::Lookback(config) => {
        let has_oob_lookbacks = delta::decode_with_lookbacks_in_place(
          config,
          delta_latents
            .unwrap()
            .downcast_ref::<DeltaLookback>()
            .unwrap(),
          &mut self.state.delta_state_pos,
          &mut self.state.delta_state,
          dst,
        );
        if has_oob_lookbacks {
          Err(PcoError::corruption(
            "delta lookback exceeded window n",
          ))
        } else {
          Ok(())
        }
      }
    }
  }
}
