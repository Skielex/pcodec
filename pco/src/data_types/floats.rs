use std::mem;

use half::f16;
use num_traits::{AsPrimitive, Float};

use crate::constants::Bitlen;
use crate::data_types::{split_latents_classic, FloatLike, Latent, NumberLike};
use crate::describers::LatentDescriber;
use crate::errors::{PcoError, PcoResult};
use crate::{
  describers, float_mult_utils, float_quant_utils, ChunkConfig, ChunkMeta, FloatMultSpec,
  FloatQuantSpec, Mode,
};

fn choose_mode_and_split_latents<F: FloatLike>(
  nums: &[F],
  chunk_config: &ChunkConfig,
) -> PcoResult<(Mode<F::L>, Vec<Vec<F::L>>)> {
  if chunk_config.float_mult_spec != FloatMultSpec::Disabled
    && chunk_config.float_quant_spec != FloatQuantSpec::Disabled
  {
    return Err(PcoError::invalid_argument(
      "FloatMult and FloatQuant cannot be used simultaneously",
    ));
  }
  Ok(
    match (
      chunk_config.float_mult_spec,
      chunk_config.float_quant_spec,
    ) {
      (FloatMultSpec::Enabled, _) => {
        if let Some(fm_config) = float_mult_utils::choose_config(nums) {
          let mode = Mode::float_mult(fm_config.base);
          let latents = float_mult_utils::split_latents(nums, fm_config.base, fm_config.inv_base);
          (mode, latents)
        } else {
          (Mode::Classic, split_latents_classic(nums))
        }
      }
      (FloatMultSpec::Provided(base_f64), _) => {
        let base: F = F::from_f64(base_f64).unwrap();
        let mode = Mode::float_mult(base);
        let latents = float_mult_utils::split_latents(nums, base, base.recip());
        (mode, latents)
      }
      (FloatMultSpec::Disabled, FloatQuantSpec::Provided(k)) => (
        Mode::FloatQuant(k),
        float_quant_utils::split_latents(nums, k),
      ),
      (FloatMultSpec::Disabled, FloatQuantSpec::Disabled) => {
        (Mode::Classic, split_latents_classic(nums))
      } // TODO(https://github.com/mwlon/pcodec/issues/194): Add a case for FloatQuantSpec::Enabled
        // once it exists
    },
  )
}

macro_rules! impl_float_like {
  ($t: ty, $latent: ty, $bits: expr, $exp_offset: expr, $zero: expr, $max_for_sampling: expr) => {
    impl FloatLike for $t {
      const BITS: Bitlen = $bits;
      /// Number of bits in the representation of the significand, excluding the implicit
      /// leading bit.  (In Rust, `MANTISSA_DIGITS` does include the implicit leading bit.)
      const PRECISION_BITS: Bitlen = Self::MANTISSA_DIGITS as Bitlen - 1;
      const ZERO: Self = $zero;
      const MAX_FOR_SAMPLING: Self = $max_for_sampling;

      #[inline]
      fn exp2_int(power: i32) -> Self {
        Self::exp2(power.as_())
      }

      #[inline]
      fn is_finite_and_normal(&self) -> bool {
        self.is_finite() && self.is_normal()
      }

      #[inline]
      fn exponent(&self) -> i32 {
        (self.abs().to_bits() >> Self::PRECISION_BITS) as i32 + $exp_offset
      }

      #[inline]
      fn trailing_zeros(&self) -> u32 {
        self.to_bits().trailing_zeros()
      }

      #[inline]
      fn to_latent_bits(self) -> Self::L {
        self.to_bits()
      }

      #[inline]
      fn int_float_from_latent(l: Self::L) -> Self {
        let mid = Self::L::MID;
        let (negative, abs_int) = if l >= mid {
          (false, l - mid)
        } else {
          (true, mid - 1 - l)
        };
        let gpi = 1 << Self::MANTISSA_DIGITS;
        let abs_float = if abs_int < gpi {
          abs_int.as_()
        } else {
          let gpi_float: Self = gpi.as_();
          Self::from_bits((gpi_float).to_bits() + (abs_int - gpi))
        };
        if negative {
          -abs_float
        } else {
          abs_float
        }
      }

      #[inline]
      fn int_float_to_latent(self) -> Self::L {
        let abs = self.abs();
        let gpi = 1 << Self::MANTISSA_DIGITS;
        let gpi_float: Self = gpi.as_();
        let abs_int: Self::L = if abs < gpi_float {
          abs.as_()
        } else {
          gpi + (abs.to_bits() - gpi_float.to_bits())
        };
        if self.is_sign_positive() {
          Self::L::MID + abs_int
        } else {
          // -1 because we need to distinguish -0.0 from +0.0
          Self::L::MID - 1 - abs_int
        }
      }

      #[inline]
      fn from_latent_numerical(l: Self::L) -> Self {
        l.as_()
      }
    }
  };
}

macro_rules! impl_float_number_like {
  ($t: ty, $latent: ty, $sign_bit_mask: expr, $header_byte: expr) => {
    impl NumberLike for $t {
      const DTYPE_BYTE: u8 = $header_byte;
      const TRANSMUTABLE_TO_LATENT: bool = true;

      type L = $latent;

      fn get_latent_describers(meta: &ChunkMeta<Self::L>) -> Vec<LatentDescriber<Self::L>> {
        describers::match_classic_mode::<Self>(meta, " ULPs")
          .or_else(|| describers::match_float_modes::<Self>(meta))
          .expect("invalid mode for float type")
      }

      fn mode_is_valid(mode: Mode<Self::L>) -> bool {
        match mode {
          Mode::Classic => true,
          Mode::FloatMult(base_latent) => {
            Self::from_latent_ordered(base_latent).is_finite_and_normal()
          }
          Mode::FloatQuant(k) => k <= Self::PRECISION_BITS,
          _ => false,
        }
      }
      fn choose_mode_and_split_latents(
        nums: &[Self],
        config: &ChunkConfig,
      ) -> (Mode<Self::L>, Vec<Vec<Self::L>>) {
        choose_mode_and_split_latents(nums, config).unwrap()
      }

      #[inline]
      fn from_latent_ordered(l: Self::L) -> Self {
        if l & $sign_bit_mask > 0 {
          // positive float
          Self::from_bits(l ^ $sign_bit_mask)
        } else {
          // negative float
          Self::from_bits(!l)
        }
      }
      #[inline]
      fn to_latent_ordered(self) -> Self::L {
        let mem_layout = self.to_bits();
        if mem_layout & $sign_bit_mask > 0 {
          // negative float
          !mem_layout
        } else {
          // positive float
          mem_layout ^ $sign_bit_mask
        }
      }
      fn join_latents(mode: Mode<Self::L>, primary: &mut [Self::L], secondary: &[Self::L]) {
        match mode {
          Mode::Classic => (),
          Mode::FloatMult(base_latent) => {
            let base = Self::from_latent_ordered(base_latent);
            float_mult_utils::join_latents(base, primary, secondary)
          }
          Mode::FloatQuant(k) => float_quant_utils::join_latents::<Self>(k, primary, secondary),
          _ => unreachable!("impossible mode for floats"),
        }
      }

      fn transmute_to_latents(slice: &mut [Self]) -> &mut [Self::L] {
        unsafe { mem::transmute(slice) }
      }

      #[inline]
      fn transmute_to_latent(self) -> Self::L {
        self.to_bits()
      }
    }
  };
}

impl_float_like!(f32, u32, 32, -127, 0.0, f32::MAX * 0.5);
impl_float_like!(f64, u64, 64, -1023, 0.0, f64::MAX * 0.5);
impl_float_like!(
  f16,
  u16,
  16,
  -15,
  f16::ZERO,
  f16::from_bits(30719)
);
// f16 FloatLike is implemented separately because it's non-native.
impl_float_number_like!(f32, u32, 1_u32 << 31, 5);
impl_float_number_like!(f64, u64, 1_u64 << 63, 6);
impl_float_number_like!(f16, u16, 1_u16 << 15, 9);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_float_ordering() {
    assert!(f32::NEG_INFINITY.to_latent_ordered() < (-0.0_f32).to_latent_ordered());
    assert!((-0.0_f32).to_latent_ordered() < (0.0_f32).to_latent_ordered());
    assert!((0.0_f32).to_latent_ordered() < f32::INFINITY.to_latent_ordered());
  }

  #[test]
  fn test_exp() {
    assert_eq!(1.0_f32.exponent(), 0);
    assert_eq!(1.0_f64.exponent(), 0);
    assert_eq!(2.0_f32.exponent(), 1);
    assert_eq!(3.3333_f32.exponent(), 1);
    assert_eq!(0.3333_f32.exponent(), -2);
    assert_eq!(31.0_f32.exponent(), 4);
  }

  #[test]
  fn int_float32_invertibility() {
    for x in [
      -f32::NAN,
      f32::NEG_INFINITY,
      f32::MIN,
      -1.0,
      -0.0,
      0.0,
      3.0,
      f32::MAX,
      f32::INFINITY,
      f32::NAN,
    ] {
      let int = x.int_float_to_latent();
      let recovered = f32::int_float_from_latent(int);
      // gotta compare unsigneds because floats don't implement Equal
      assert_eq!(
        x.to_bits(),
        recovered.to_bits(),
        "{} != {}",
        x,
        recovered
      );
    }
  }

  #[test]
  fn int_float64_invertibility() {
    for x in [
      -f64::NAN,
      f64::NEG_INFINITY,
      f64::MIN,
      -1.0,
      -0.0,
      0.0,
      3.0,
      f64::MAX,
      f64::INFINITY,
      f64::NAN,
    ] {
      let int = x.int_float_to_latent();
      let recovered = f64::int_float_from_latent(int);
      // gotta compare unsigneds because floats don't implement Equal
      assert_eq!(
        x.to_bits(),
        recovered.to_bits(),
        "{} != {}",
        x,
        recovered
      );
    }
  }

  #[test]
  fn int_float_ordering() {
    let values = vec![
      -f32::NAN,
      f32::NEG_INFINITY,
      f32::MIN,
      -1.0,
      -0.0,
      0.0,
      3.0,
      (1 << 24) as f32,
      f32::MAX,
      f32::INFINITY,
      f32::NAN,
    ];
    let mut last_int = None;
    for x in values {
      let int = x.int_float_to_latent();
      if let Some(last_int) = last_int {
        assert!(
          last_int < int,
          "at {}; int {} vs {}",
          x,
          last_int,
          int
        );
      }
      last_int = Some(int)
    }
  }
}
