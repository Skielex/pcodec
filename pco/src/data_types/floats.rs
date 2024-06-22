use std::mem::{self, transmute};

use half::f16;

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
        let base = F::from_f64(base_f64);
        let mode = Mode::float_mult(base);
        let latents = float_mult_utils::split_latents(nums, base, base.inv());
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
  ($t: ty, $latent: ty, $bits: expr, $exp_offset: expr, $exp2_lut: expr) => {
    impl FloatLike for $t {
      const BITS: Bitlen = $bits;
      /// Number of bits in the representation of the significand, excluding the implicit
      /// leading bit.  (In Rust, `MANTISSA_DIGITS` does include the implicit leading bit.)
      const PRECISION_BITS: Bitlen = Self::MANTISSA_DIGITS as Bitlen - 1;
      const ZERO: Self = 0.0;
      const MAX_FOR_SAMPLING: Self = Self::MAX * 0.5;
      const EXP2_LUT: [Self; 128] = $exp2_lut;

      #[inline]
      fn abs(self) -> Self {
        self.abs()
      }

      fn inv(self) -> Self {
        1.0 / self
      }

      #[inline]
      fn round(self) -> Self {
        self.round()
      }

      #[inline]
      fn exp2(power: i32) -> Self {
        // Self::exp2(power as Self)
        Self::EXP2_LUT[(power + 64) as usize]
      }

      #[inline]
      fn from_f64(x: f64) -> Self {
        x as Self
      }

      #[inline]
      fn to_f64(self) -> f64 {
        self as f64
      }

      #[inline]
      fn is_finite_and_normal(&self) -> bool {
        self.is_finite() && !self.is_subnormal()
      }

      #[inline]
      fn is_sign_positive_(&self) -> bool {
        self.is_sign_positive()
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
      fn max(a: Self, b: Self) -> Self {
        Self::max(a, b)
      }

      #[inline]
      fn min(a: Self, b: Self) -> Self {
        Self::min(a, b)
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
          abs_int as Self
        } else {
          Self::from_bits((gpi as Self).to_bits() + (abs_int - gpi))
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
        let gpi_float = gpi as Self;
        let abs_int = if abs < gpi_float {
          abs as Self::L
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
        l as Self
      }
    }
  };
}

impl FloatLike for f16 {
  const BITS: Bitlen = 16;
  const PRECISION_BITS: Bitlen = Self::MANTISSA_DIGITS as Bitlen - 1;
  const ZERO: Self = f16::ZERO;
  const MAX_FOR_SAMPLING: Self = f16::from_bits(30719); // Half of MAX size.
  const EXP2_LUT: [Self; 128] = [
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(0),
    f16::from_bits(1),
    f16::from_bits(2),
    f16::from_bits(4),
    f16::from_bits(8),
    f16::from_bits(16),
    f16::from_bits(32),
    f16::from_bits(64),
    f16::from_bits(128),
    f16::from_bits(256),
    f16::from_bits(512),
    f16::from_bits(1024),
    f16::from_bits(2048),
    f16::from_bits(3072),
    f16::from_bits(4096),
    f16::from_bits(5120),
    f16::from_bits(6144),
    f16::from_bits(7168),
    f16::from_bits(8192),
    f16::from_bits(9216),
    f16::from_bits(10240),
    f16::from_bits(11264),
    f16::from_bits(12288),
    f16::from_bits(13312),
    f16::from_bits(14336),
    f16::from_bits(15360),
    f16::from_bits(16384),
    f16::from_bits(17408),
    f16::from_bits(18432),
    f16::from_bits(19456),
    f16::from_bits(20480),
    f16::from_bits(21504),
    f16::from_bits(22528),
    f16::from_bits(23552),
    f16::from_bits(24576),
    f16::from_bits(25600),
    f16::from_bits(26624),
    f16::from_bits(27648),
    f16::from_bits(28672),
    f16::from_bits(29696),
    f16::from_bits(30720),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
    f16::from_bits(31744),
  ];

  #[inline]
  fn abs(self) -> Self {
    Self::from_bits(self.to_bits() & 0x7FFF)
  }

  fn inv(self) -> Self {
    Self::ONE / self
  }

  #[inline]
  fn round(self) -> Self {
    Self::from_f32(self.to_f32().round())
  }

  #[inline]
  fn exp2(power: i32) -> Self {
    Self::EXP2_LUT[(power + 64) as usize]
  }

  #[inline]
  fn from_f64(x: f64) -> Self {
    Self::from_f64(x)
  }

  #[inline]
  fn to_f64(self) -> f64 {
    self.to_f64()
  }

  #[inline]
  fn is_finite_and_normal(&self) -> bool {
    self.is_finite() && self.is_normal()
  }

  #[inline]
  fn is_sign_positive_(&self) -> bool {
    self.is_sign_positive()
  }

  #[inline]
  fn exponent(&self) -> i32 {
    (self.abs().to_bits() >> Self::PRECISION_BITS) as i32 - 15
  }

  #[inline]
  fn trailing_zeros(&self) -> u32 {
    self.to_bits().trailing_zeros()
  }

  #[inline]
  fn max(a: Self, b: Self) -> Self {
    Self::max(a, b)
  }

  #[inline]
  fn min(a: Self, b: Self) -> Self {
    Self::min(a, b)
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
      Self::from_f32(abs_int as f32)
    } else {
      Self::from_bits(Self::from_f32(gpi as f32).to_bits() + (abs_int - gpi))
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
    let gpi_float = Self::from_f32(gpi as f32);
    let abs_int = if abs < gpi_float {
      abs.to_f32() as Self::L
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
    Self::from_f32(l as f32)
  }
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
const F64_LUT: [f64; 128] = {
  unsafe {
    [
      transmute::<u64, f64>(4318952042648305664),
      transmute::<u64, f64>(4323455642275676160),
      transmute::<u64, f64>(4327959241903046656),
      transmute::<u64, f64>(4332462841530417152),
      transmute::<u64, f64>(4336966441157787648),
      transmute::<u64, f64>(4341470040785158144),
      transmute::<u64, f64>(4345973640412528640),
      transmute::<u64, f64>(4350477240039899136),
      transmute::<u64, f64>(4354980839667269632),
      transmute::<u64, f64>(4359484439294640128),
      transmute::<u64, f64>(4363988038922010624),
      transmute::<u64, f64>(4368491638549381120),
      transmute::<u64, f64>(4372995238176751616),
      transmute::<u64, f64>(4377498837804122112),
      transmute::<u64, f64>(4382002437431492608),
      transmute::<u64, f64>(4386506037058863104),
      transmute::<u64, f64>(4391009636686233600),
      transmute::<u64, f64>(4395513236313604096),
      transmute::<u64, f64>(4400016835940974592),
      transmute::<u64, f64>(4404520435568345088),
      transmute::<u64, f64>(4409024035195715584),
      transmute::<u64, f64>(4413527634823086080),
      transmute::<u64, f64>(4418031234450456576),
      transmute::<u64, f64>(4422534834077827072),
      transmute::<u64, f64>(4427038433705197568),
      transmute::<u64, f64>(4431542033332568064),
      transmute::<u64, f64>(4436045632959938560),
      transmute::<u64, f64>(4440549232587309056),
      transmute::<u64, f64>(4445052832214679552),
      transmute::<u64, f64>(4449556431842050048),
      transmute::<u64, f64>(4454060031469420544),
      transmute::<u64, f64>(4458563631096791040),
      transmute::<u64, f64>(4463067230724161536),
      transmute::<u64, f64>(4467570830351532032),
      transmute::<u64, f64>(4472074429978902528),
      transmute::<u64, f64>(4476578029606273024),
      transmute::<u64, f64>(4481081629233643520),
      transmute::<u64, f64>(4485585228861014016),
      transmute::<u64, f64>(4490088828488384512),
      transmute::<u64, f64>(4494592428115755008),
      transmute::<u64, f64>(4499096027743125504),
      transmute::<u64, f64>(4503599627370496000),
      transmute::<u64, f64>(4508103226997866496),
      transmute::<u64, f64>(4512606826625236992),
      transmute::<u64, f64>(4517110426252607488),
      transmute::<u64, f64>(4521614025879977984),
      transmute::<u64, f64>(4526117625507348480),
      transmute::<u64, f64>(4530621225134718976),
      transmute::<u64, f64>(4535124824762089472),
      transmute::<u64, f64>(4539628424389459968),
      transmute::<u64, f64>(4544132024016830464),
      transmute::<u64, f64>(4548635623644200960),
      transmute::<u64, f64>(4553139223271571456),
      transmute::<u64, f64>(4557642822898941952),
      transmute::<u64, f64>(4562146422526312448),
      transmute::<u64, f64>(4566650022153682944),
      transmute::<u64, f64>(4571153621781053440),
      transmute::<u64, f64>(4575657221408423936),
      transmute::<u64, f64>(4580160821035794432),
      transmute::<u64, f64>(4584664420663164928),
      transmute::<u64, f64>(4589168020290535424),
      transmute::<u64, f64>(4593671619917905920),
      transmute::<u64, f64>(4598175219545276416),
      transmute::<u64, f64>(4602678819172646912),
      transmute::<u64, f64>(4607182418800017408),
      transmute::<u64, f64>(4611686018427387904),
      transmute::<u64, f64>(4616189618054758400),
      transmute::<u64, f64>(4620693217682128896),
      transmute::<u64, f64>(4625196817309499392),
      transmute::<u64, f64>(4629700416936869888),
      transmute::<u64, f64>(4634204016564240384),
      transmute::<u64, f64>(4638707616191610880),
      transmute::<u64, f64>(4643211215818981376),
      transmute::<u64, f64>(4647714815446351872),
      transmute::<u64, f64>(4652218415073722368),
      transmute::<u64, f64>(4656722014701092864),
      transmute::<u64, f64>(4661225614328463360),
      transmute::<u64, f64>(4665729213955833856),
      transmute::<u64, f64>(4670232813583204352),
      transmute::<u64, f64>(4674736413210574848),
      transmute::<u64, f64>(4679240012837945344),
      transmute::<u64, f64>(4683743612465315840),
      transmute::<u64, f64>(4688247212092686336),
      transmute::<u64, f64>(4692750811720056832),
      transmute::<u64, f64>(4697254411347427328),
      transmute::<u64, f64>(4701758010974797824),
      transmute::<u64, f64>(4706261610602168320),
      transmute::<u64, f64>(4710765210229538816),
      transmute::<u64, f64>(4715268809856909312),
      transmute::<u64, f64>(4719772409484279808),
      transmute::<u64, f64>(4724276009111650304),
      transmute::<u64, f64>(4728779608739020800),
      transmute::<u64, f64>(4733283208366391296),
      transmute::<u64, f64>(4737786807993761792),
      transmute::<u64, f64>(4742290407621132288),
      transmute::<u64, f64>(4746794007248502784),
      transmute::<u64, f64>(4751297606875873280),
      transmute::<u64, f64>(4755801206503243776),
      transmute::<u64, f64>(4760304806130614272),
      transmute::<u64, f64>(4764808405757984768),
      transmute::<u64, f64>(4769312005385355264),
      transmute::<u64, f64>(4773815605012725760),
      transmute::<u64, f64>(4778319204640096256),
      transmute::<u64, f64>(4782822804267466752),
      transmute::<u64, f64>(4787326403894837248),
      transmute::<u64, f64>(4791830003522207744),
      transmute::<u64, f64>(4796333603149578240),
      transmute::<u64, f64>(4800837202776948736),
      transmute::<u64, f64>(4805340802404319232),
      transmute::<u64, f64>(4809844402031689728),
      transmute::<u64, f64>(4814348001659060224),
      transmute::<u64, f64>(4818851601286430720),
      transmute::<u64, f64>(4823355200913801216),
      transmute::<u64, f64>(4827858800541171712),
      transmute::<u64, f64>(4832362400168542208),
      transmute::<u64, f64>(4836865999795912704),
      transmute::<u64, f64>(4841369599423283200),
      transmute::<u64, f64>(4845873199050653696),
      transmute::<u64, f64>(4850376798678024192),
      transmute::<u64, f64>(4854880398305394688),
      transmute::<u64, f64>(4859383997932765184),
      transmute::<u64, f64>(4863887597560135680),
      transmute::<u64, f64>(4868391197187506176),
      transmute::<u64, f64>(4872894796814876672),
      transmute::<u64, f64>(4877398396442247168),
      transmute::<u64, f64>(4881901996069617664),
      transmute::<u64, f64>(4886405595696988160),
      transmute::<u64, f64>(4890909195324358656),
    ]
  }
};

const F32_LUT: [f32; 128] = {
  unsafe {
    [
      transmute::<u32, f32>(528482304),
      transmute::<u32, f32>(536870912),
      transmute::<u32, f32>(545259520),
      transmute::<u32, f32>(553648128),
      transmute::<u32, f32>(562036736),
      transmute::<u32, f32>(570425344),
      transmute::<u32, f32>(578813952),
      transmute::<u32, f32>(587202560),
      transmute::<u32, f32>(595591168),
      transmute::<u32, f32>(603979776),
      transmute::<u32, f32>(612368384),
      transmute::<u32, f32>(620756992),
      transmute::<u32, f32>(629145600),
      transmute::<u32, f32>(637534208),
      transmute::<u32, f32>(645922816),
      transmute::<u32, f32>(654311424),
      transmute::<u32, f32>(662700032),
      transmute::<u32, f32>(671088640),
      transmute::<u32, f32>(679477248),
      transmute::<u32, f32>(687865856),
      transmute::<u32, f32>(696254464),
      transmute::<u32, f32>(704643072),
      transmute::<u32, f32>(713031680),
      transmute::<u32, f32>(721420288),
      transmute::<u32, f32>(729808896),
      transmute::<u32, f32>(738197504),
      transmute::<u32, f32>(746586112),
      transmute::<u32, f32>(754974720),
      transmute::<u32, f32>(763363328),
      transmute::<u32, f32>(771751936),
      transmute::<u32, f32>(780140544),
      transmute::<u32, f32>(788529152),
      transmute::<u32, f32>(796917760),
      transmute::<u32, f32>(805306368),
      transmute::<u32, f32>(813694976),
      transmute::<u32, f32>(822083584),
      transmute::<u32, f32>(830472192),
      transmute::<u32, f32>(838860800),
      transmute::<u32, f32>(847249408),
      transmute::<u32, f32>(855638016),
      transmute::<u32, f32>(864026624),
      transmute::<u32, f32>(872415232),
      transmute::<u32, f32>(880803840),
      transmute::<u32, f32>(889192448),
      transmute::<u32, f32>(897581056),
      transmute::<u32, f32>(905969664),
      transmute::<u32, f32>(914358272),
      transmute::<u32, f32>(922746880),
      transmute::<u32, f32>(931135488),
      transmute::<u32, f32>(939524096),
      transmute::<u32, f32>(947912704),
      transmute::<u32, f32>(956301312),
      transmute::<u32, f32>(964689920),
      transmute::<u32, f32>(973078528),
      transmute::<u32, f32>(981467136),
      transmute::<u32, f32>(989855744),
      transmute::<u32, f32>(998244352),
      transmute::<u32, f32>(1006632960),
      transmute::<u32, f32>(1015021568),
      transmute::<u32, f32>(1023410176),
      transmute::<u32, f32>(1031798784),
      transmute::<u32, f32>(1040187392),
      transmute::<u32, f32>(1048576000),
      transmute::<u32, f32>(1056964608),
      transmute::<u32, f32>(1065353216),
      transmute::<u32, f32>(1073741824),
      transmute::<u32, f32>(1082130432),
      transmute::<u32, f32>(1090519040),
      transmute::<u32, f32>(1098907648),
      transmute::<u32, f32>(1107296256),
      transmute::<u32, f32>(1115684864),
      transmute::<u32, f32>(1124073472),
      transmute::<u32, f32>(1132462080),
      transmute::<u32, f32>(1140850688),
      transmute::<u32, f32>(1149239296),
      transmute::<u32, f32>(1157627904),
      transmute::<u32, f32>(1166016512),
      transmute::<u32, f32>(1174405120),
      transmute::<u32, f32>(1182793728),
      transmute::<u32, f32>(1191182336),
      transmute::<u32, f32>(1199570944),
      transmute::<u32, f32>(1207959552),
      transmute::<u32, f32>(1216348160),
      transmute::<u32, f32>(1224736768),
      transmute::<u32, f32>(1233125376),
      transmute::<u32, f32>(1241513984),
      transmute::<u32, f32>(1249902592),
      transmute::<u32, f32>(1258291200),
      transmute::<u32, f32>(1266679808),
      transmute::<u32, f32>(1275068416),
      transmute::<u32, f32>(1283457024),
      transmute::<u32, f32>(1291845632),
      transmute::<u32, f32>(1300234240),
      transmute::<u32, f32>(1308622848),
      transmute::<u32, f32>(1317011456),
      transmute::<u32, f32>(1325400064),
      transmute::<u32, f32>(1333788672),
      transmute::<u32, f32>(1342177280),
      transmute::<u32, f32>(1350565888),
      transmute::<u32, f32>(1358954496),
      transmute::<u32, f32>(1367343104),
      transmute::<u32, f32>(1375731712),
      transmute::<u32, f32>(1384120320),
      transmute::<u32, f32>(1392508928),
      transmute::<u32, f32>(1400897536),
      transmute::<u32, f32>(1409286144),
      transmute::<u32, f32>(1417674752),
      transmute::<u32, f32>(1426063360),
      transmute::<u32, f32>(1434451968),
      transmute::<u32, f32>(1442840576),
      transmute::<u32, f32>(1451229184),
      transmute::<u32, f32>(1459617792),
      transmute::<u32, f32>(1468006400),
      transmute::<u32, f32>(1476395008),
      transmute::<u32, f32>(1484783616),
      transmute::<u32, f32>(1493172224),
      transmute::<u32, f32>(1501560832),
      transmute::<u32, f32>(1509949440),
      transmute::<u32, f32>(1518338048),
      transmute::<u32, f32>(1526726656),
      transmute::<u32, f32>(1535115264),
      transmute::<u32, f32>(1543503872),
      transmute::<u32, f32>(1551892480),
      transmute::<u32, f32>(1560281088),
      transmute::<u32, f32>(1568669696),
      transmute::<u32, f32>(1577058304),
      transmute::<u32, f32>(1585446912),
      transmute::<u32, f32>(1593835520),
    ]
  }
};

impl_float_like!(f32, u32, 32, -127, F32_LUT);
impl_float_like!(f64, u64, 64, -1023, F64_LUT);
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
