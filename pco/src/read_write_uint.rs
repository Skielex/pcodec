use std::fmt::{Debug, Display};
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Shl, Shr, Sub};

use crate::constants::Bitlen;
use crate::data_types::Latent;

// this applies to reading and also works for byte-aligned precisions
pub const fn calc_max_bytes(precision: Bitlen) -> usize {
  // See bit_reader::read_uint_at for an explanation of these thresholds.
  if precision == 0 {
    0
  // } else if precision <= 9 {
  //   2
  } else if precision <= 25 {
    4
  } else if precision <= 57 {
    8
  } else if precision <= 113 {
    16
  } else {
    24
  }
}

pub const fn calc_max_bytes_for_writing(precision: Bitlen) -> usize {
  // We need to be slightly more conservative during writing
  // due to how write_short_uints is implemented.
  if precision == 0 {
    0
  } else if precision <= 24 {
    4
  } else if precision <= 56 {
    8
  } else if precision <= 113 {
    16
  } else {
    24
  }
}

pub trait ReadWriteUint:
  Add<Output = Self>
  + BitAnd<Output = Self>
  + BitOr<Output = Self>
  + BitAndAssign
  + BitOrAssign
  + Copy
  + Debug
  + Display
  + Shl<Bitlen, Output = Self>
  + Shr<Bitlen, Output = Self>
  + Sub<Output = Self>
{
  const ONE: Self;
  const BITS: Bitlen;
  const MAX_BYTES: usize = calc_max_bytes(Self::BITS);

  fn from_u32(x: u32) -> Self;
  #[allow(dead_code)]
  fn to_u32(self) -> u32;
  fn from_u64(x: u64) -> Self;
  fn to_u64(self) -> u64;
}

impl ReadWriteUint for usize {
  const ONE: Self = 1;
  const BITS: Bitlen = usize::BITS;

  #[inline]
  fn from_u32(x: u32) -> Self {
    x as Self
  }

  #[inline]
  fn to_u32(self) -> u32 {
    self as u32
  }

  #[inline]
  fn from_u64(x: u64) -> Self {
    x as Self
  }

  #[inline]
  fn to_u64(self) -> u64 {
    self as u64
  }
}

impl<L: Latent> ReadWriteUint for L {
  const ONE: Self = <Self as Latent>::ONE;
  const BITS: Bitlen = <Self as Latent>::BITS;

  #[inline]
  fn from_u32(x: u32) -> Self {
    <Self as Latent>::from_u32(x)
  }

  #[inline]
  fn to_u32(self) -> u32 {
    <Self as Latent>::to_u32(self)
  }

  #[inline]
  fn from_u64(x: u64) -> Self {
    <Self as Latent>::from_u64(x)
  }

  #[inline]
  fn to_u64(self) -> u64 {
    <Self as Latent>::to_u64(self)
  }
}
