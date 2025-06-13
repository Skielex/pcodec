use crate::ans::spec::Spec;
use crate::ans::{AnsState, CompactAnsState, CompactSymbol};
use crate::constants::CompactBitlen;

#[derive(Clone, Debug)]
pub struct Decoder {
  pub symbols: Vec<CompactSymbol>,
  pub next_state_idx_bases: Vec<CompactAnsState>,
  pub bits_to_reads: Vec<CompactBitlen>,
}
impl Decoder {
  pub fn new(spec: &Spec) -> Self {
    let table_size = spec.table_size();
    let mut symbols = Vec::with_capacity(table_size);
    let mut next_state_idx_bases = Vec::with_capacity(table_size);
    let mut bits_to_reads = Vec::with_capacity(table_size);

    // x_s from Jarek Duda's paper
    let mut symbol_x_s = spec.symbol_weights.clone();
    for &symbol in &spec.state_symbols {
      let next_state_base = symbol_x_s[symbol as usize] as AnsState;
      let bits_to_read = next_state_base.leading_zeros() - (table_size as AnsState).leading_zeros();
      let next_state_base = next_state_base << bits_to_read;

      symbols.push(symbol as CompactSymbol);
      next_state_idx_bases.push((next_state_base - table_size as AnsState) as CompactAnsState);
      bits_to_reads.push(bits_to_read as CompactBitlen);
      symbol_x_s[symbol as usize] += 1;
    }

    Self {
      symbols,
      next_state_idx_bases,
      bits_to_reads,
    }
  }
}
