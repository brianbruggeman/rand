use std::any::TypeId;
use std::iter::Sum;
use std::ops::{Bound, Div, RangeBounds, Sub};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use num_traits::{AsPrimitive, Bounded, ToPrimitive, WrappingAdd};

#[cfg(target_pointer_width = "16")]
mod data {
    pub type BaseType = u16;
    pub type SeedType = u16;
    pub const BITS: [SeedType; 8] = [
        0xA7F9, // 0b1010'0111'1111'1001
        0xBEEF, // 0b1011'1110'1110'1111
        0xC0FF, // 0b1100'0000'1111'1111
        0xD3B7, // 0b1101'0011'1011'0111
        0xDEAD, // 0b1101'1110'1010'1101
        0xE5B7, // 0b1110'0101'1011'0111
        0xFACE, // 0b1111'1010'1100'1110
        0xF00D, // 0b1111'0000'0000'1101
    ];
    pub const SEED: SeedType = 0xD3B7;
    pub const SHIFTS: [u32; 8] = [8, 8, 8, 8, 8, 8, 8, 8];
}

#[cfg(target_pointer_width = "32")]
mod data {
    pub type BaseType = u32;
    pub type SeedType = u32;
    pub const BITS: [SeedType; 8] = [
        0x68E31DA4, // 0b0110'1000'1110'0011'0001'1101'1010'0100
        0xB5297A4D, // 0b1011'0101'0010'1001'0111'1010'0100'1101
        0x1B56C4E9, // 0b0001'1011'0101'0110'1100'1100'1110'1001
        0xA37B4539, // 0b1010'0011'0111'1011'0100'0101'0011'1001
        0x72BE5D74, // 0b0111'0010'1011'1110'0101'1101'0111'0100
        0xC3E1F763, // 0b1100'0011'1110'0001'1111'0111'0110'0011
        0xD0B3AC93, // 0b1101'0000'1011'0011'1010'1100'1001'0011
        0x9ACFC8C5, // 0b1001'1010'1100'1111'1100'1000'1100'0101
    ];
    pub const SEED: SeedType = 0xA37B4539;
    pub const SHIFTS: [u32; 8] = [23, 19, 17, 11, 5, 3, 2, 1];
}

#[cfg(target_pointer_width = "64")]
mod data {
    pub type BaseType = u64;
    pub type SeedType = BaseType;
    pub const BITS: [SeedType; 8] = [
        (0x68E31DA4 as BaseType) << 32 | 0xB5297A4D as BaseType, // 0b0110'1000'1110'0011'0001'1101'1010'0100'1011'0101'0010'1001'0111'1010'0100'1101
        (0x1B56C4E9 as BaseType) << 32 | 0xA37B4539 as BaseType, // 0b0001'1011'0101'0110'1100'1100'1110'1001'1010'0011'0111'1011'0100'0101'0011'1001
        (0x72BE5D74 as BaseType) << 32 | 0xC3E1F763 as BaseType, // 0b0111'0010'1011'1110'0101'1101'0111'0100'1100'0011'1110'0001'1111'0111'0110'0011
        (0xD0B3AC93 as BaseType) << 32 | 0x9ACFC8C5 as BaseType, // 0b1101'0000'1011'0011'1010'1100'1001'0011'1001'1010'1100'1111'1100'1000'1100'0101
        0xFFFFFFFFFFFFFFC5,                                      // 0b1111'1111'1111'1111'1111'1111'1111'1101
        0xFFFFFFFFFFFFFF43,                                      // 0b1111'1111'1111'1111'1111'1111'1111'0100
        0xFFFFFFFFFFFFFC2F,                                      // 0b1111'1111'1111'1111'1111'1111'1100'1111
        0xFFFFFFFFFFFFF837,                                      // 0b1111'1111'1111'1111'1111'1111'1100'1000
    ];
    pub const SEED: SeedType = (0xD0B3AC93 as BaseType) << 32 | 0x9ACFC8C5;
    pub const SHIFTS: [u32; 8] = [41, 37, 29, 23, 19, 17, 11, 7];
}

type BaseType = data::BaseType;
type SeedType = data::SeedType;

const BASE_TYPE_MAX_F64: f64 = BaseType::MAX as f64;
const SEED: SeedType = data::SEED;
const BITS: [SeedType; 8] = data::BITS;
const SHIFTS: [u32; 8] = data::SHIFTS;

#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    pub state: f64,
    seed: SeedType,
}

pub trait NoiseVec {
    fn with_noise(size: impl AsPrimitive<usize>) -> Self;
    fn with_random_noise(size: impl AsPrimitive<usize>) -> Self;
    fn with_seeded_noise(size: impl AsPrimitive<usize>, seed: impl AsPrimitive<BaseType>) -> Self;
    fn with_random_seeded_noise(size: impl AsPrimitive<usize>, seed: impl AsPrimitive<BaseType>) -> Self;
}

impl<O> NoiseVec for Vec<O>
where
    O: AsPrimitive<BaseType> + ToPrimitive + Bounded + Send + Div<Output = O> + Sub<Output = O>,
    f64: AsPrimitive<O>,
    usize: AsPrimitive<O>,
    BaseType: AsPrimitive<O>,
{
    fn with_noise(size: impl AsPrimitive<usize>) -> Self {
        Self::with_seeded_noise(size, SEED)
    }

    fn with_seeded_noise(count: impl AsPrimitive<usize>, seed: impl AsPrimitive<SeedType>) -> Self {
        let seed = seed.as_();
        let mut vec = Vec::with_capacity(count.as_());
        vec.par_extend((0..count.as_()).into_par_iter().map(|idx| {
            if TypeId::of::<O>() == TypeId::of::<f64>() || TypeId::of::<O>() == TypeId::of::<f32>() {
                let noise = get_1d_noise(idx, seed, BITS, SHIFTS).to_f64().unwrap();
                let max = BaseType::MAX.to_f64().unwrap();
                noise.div(max).as_()
            } else {
                get_1d_noise(idx, seed, BITS, SHIFTS)
            }
        }));
        vec
    }

    fn with_random_noise(count: impl AsPrimitive<usize>) -> Self {
        Self::with_random_seeded_noise(count, SEED)
    }

    fn with_random_seeded_noise(count: impl AsPrimitive<usize>, seed: impl AsPrimitive<SeedType>) -> Self {
        let seed = seed.as_();
        let mut vec = Vec::with_capacity(count.as_());
        let chunk_size = count.as_() / rayon::current_num_threads();

        vec.par_chunks_mut(chunk_size).for_each(|chunk| {
            let mut rng = Rng::from_seed(seed);
            chunk.iter_mut().for_each(|element| {
                *element = rng.gen::<O>();
            });
        });
        vec
    }
}

impl Default for Rng {
    fn default() -> Self {
        // Start with something random
        let value = Self::initital_state();
        Self {
            state: value.as_(),
            seed: get_fast_1d_noise(value, SEED),
        }
    }
}

impl Rng {
    pub fn new(seed: impl AsPrimitive<u64>) -> Self {
        let mut new_rng = Self { state: 0.as_(), seed: seed.as_() };
        new_rng.rand();
        new_rng
    }

    pub fn state(&self) -> f64 {
        self.state
    }

    pub fn seed(&self) -> SeedType {
        self.seed
    }

    pub fn initital_state() -> SeedType {
        let start = SystemTime::now();
        let since_epoch = start.duration_since(UNIX_EPOCH).unwrap();
        let nanos: SeedType = since_epoch.subsec_nanos().as_();
        let secs: SeedType = since_epoch.as_secs().as_();
        secs ^ nanos
    }

    pub fn from_seed(seed: impl AsPrimitive<SeedType>) -> Self {
        let mut new_rng = Self::new(seed);
        new_rng.seed = seed.as_();
        new_rng
    }

    pub fn with_seed(&mut self, seed: impl AsPrimitive<SeedType>) -> &mut Self {
        self.seed = seed.as_();
        self
    }

    #[inline(always)]
    pub fn rand(&mut self) -> f64 {
        self.state = self.random_from(self.state);
        self.state / BASE_TYPE_MAX_F64
    }

    pub fn gen<O>(&mut self) -> O
    where
        O: AsPrimitive<BaseType> + Bounded,
        f64: AsPrimitive<O>,
        BaseType: AsPrimitive<O>,
    {
        if TypeId::of::<O>() == TypeId::of::<f64>() || TypeId::of::<O>() == TypeId::of::<f32>() {
            self.rand().as_()
        } else {
            self.random()
        }
    }

    #[inline(always)]
    pub fn rand_from(&self, value: impl AsPrimitive<f64>) -> f64 {
        self.random_from::<f64>(value) / BASE_TYPE_MAX_F64
    }

    #[inline(always)]
    pub fn random<O>(&mut self) -> O
    where
        O: AsPrimitive<BaseType> + Bounded,
        f64: AsPrimitive<O>,
        BaseType: AsPrimitive<O>,
    {
        self.rand(); // update state
        self.random_from(self.state)
    }

    #[inline(always)]
    pub fn random_from<O>(&self, value: impl AsPrimitive<f64>) -> O
    where
        O: AsPrimitive<BaseType> + Bounded,
        f64: AsPrimitive<O>,
        BaseType: AsPrimitive<O>,
    {
        get_fast_1d_noise(value.as_(), self.seed)
    }

    #[inline]
    pub fn random_between<O>(&mut self, min: impl AsPrimitive<f64>, max: impl AsPrimitive<f64>) -> O
    where
        O: AsPrimitive<BaseType>,
        f64: AsPrimitive<O>,
    {
        let min = min.as_();
        let max = max.as_();
        let range = max - min;
        let value = self.rand() * range;
        (value + min).as_()
    }

    #[inline(always)]
    pub fn gen_range<O>(&mut self, range: impl RangeBounds<O>) -> O
    where
        f64: AsPrimitive<O> + ToPrimitive,
        O: Copy + ToPrimitive + 'static + Bounded,
    {
        let default_max = O::max_value().to_f64().unwrap_or_default().min(BASE_TYPE_MAX_F64);
        let start = match range.start_bound() {
            Bound::Included(start) => start.to_f64().unwrap_or_default(),
            Bound::Excluded(start) => start.to_f64().unwrap_or_default() + f64::EPSILON,
            Bound::Unbounded => 0.0, // Default for unbounded start
        };

        let end = match range.end_bound() {
            Bound::Included(end) => end.to_f64().unwrap_or(default_max),
            Bound::Excluded(end) => end.to_f64().unwrap_or(default_max) - f64::EPSILON,
            Bound::Unbounded => default_max,
        };

        // Compute the range size
        let range_size = (end - start).abs();

        // Generate the random value in the calculated range
        let rnd = self.rand();
        let value = start + (rnd * range_size);

        // Convert back to the output type
        value.min(default_max).as_()
    }

    pub fn choose<'a, O>(&mut self, mut data: impl Iterator<Item = &'a O>) -> Option<&'a O> {
        match data.size_hint() {
            (len, Some(upper_bound)) if len == upper_bound => data.nth(self.gen_range(0..len)),

            // Fallback: reservoir sampling for unknown-size iterators
            _ => data
                .enumerate()
                .fold(None, |chosen: Option<&'a O>, (count, item)| match self.gen_range(0..=count) == 0 {
                    true => Some(item),
                    false => chosen,
                }),
        }
    }
}

pub fn thread_rng() -> Rng {
    Rng::default()
}

pub fn rand() -> f64 {
    let value = Rng::initital_state();
    let seed = Rng::initital_state();
    get_fast_1d_noise(value, seed)
}

pub fn get_noise<V, S, O>(position: impl IntoIterator<Item = V>, seed: S) -> O
where
    V: AsPrimitive<BaseType>,
    S: AsPrimitive<BaseType>,
    O: AsPrimitive<BaseType> + Sum<O> + WrappingAdd,
    BaseType: AsPrimitive<V>,
    BaseType: AsPrimitive<S>,
    BaseType: AsPrimitive<O>,
{
    position
        .into_iter()
        .zip(BITS.iter())
        .map(|(value, &bit)| {
            let noise: O = get_1d_noise(value, seed, BITS, SHIFTS);
            let bit = bit.as_();
            noise.as_().wrapping_mul(bit)
        })
        .fold(0 as BaseType, |acc: BaseType, x| acc.wrapping_add(x))
        .as_()
}

#[inline(always)]
pub fn get_1d_noise<X, S, O>(value: X, seed: S, bits: impl IntoIterator<Item = impl AsPrimitive<BaseType>>, shifts: impl IntoIterator<Item = impl AsPrimitive<u32>>) -> O
where
    X: AsPrimitive<BaseType>,
    S: AsPrimitive<BaseType>,
    O: Copy + 'static,
    BaseType: AsPrimitive<O>,
{
    let mut mangled_bits: BaseType = value.as_();
    for (index, (bit, shift)) in bits.into_iter().zip(shifts.into_iter()).enumerate() {
        match index % 2 == 0 {
            true => match index == 0 {
                true => {
                    mangled_bits = mangled_bits.wrapping_mul(bit.as_());
                    mangled_bits = mangled_bits.wrapping_add(seed.as_());
                    mangled_bits ^= mangled_bits.wrapping_shr(shift.as_());
                }
                false => {
                    mangled_bits = mangled_bits.wrapping_mul(bit.as_());
                    mangled_bits ^= mangled_bits.wrapping_shr(shift.as_())
                }
            },
            false => {
                mangled_bits = mangled_bits.wrapping_add(bit.as_());
                mangled_bits ^= mangled_bits.wrapping_shl(shift.as_())
            }
        }
    }
    mangled_bits.as_()
}

#[inline(always)]
pub fn get_fast_1d_noise<O>(value: impl AsPrimitive<BaseType>, seed: impl AsPrimitive<BaseType>) -> O
where
    O: AsPrimitive<BaseType>,
    BaseType: AsPrimitive<O>,
{
    get_1d_noise(value, seed, BITS.into_iter().take(3), SHIFTS.into_iter().take(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;
    use statrs::statistics::Statistics;
    use std::fmt::Debug;

    #[rstest]
    #[case::u8_0(0_u8, SEED, BITS, SHIFTS, 3_u8)]
    #[case::u8_1(1_u8, SEED, BITS, SHIFTS, 186_u8)]
    #[case::u8_20(20_u8, SEED, BITS, SHIFTS, 89_u8)]
    #[case::u16_0(0_u16, SEED, BITS, SHIFTS, 55811_u16)]
    #[case::u16_1(1_u16, SEED, BITS, SHIFTS, 18618_u16)]
    #[case::u32_0(0_u32, SEED, BITS, SHIFTS, 3023952387_u32)]
    #[case::u32_1(1_u32, SEED, BITS, SHIFTS, 2627291322_u32)]
    #[case::u64_0(0_u64, SEED, BITS, SHIFTS, 12296723819396979203_u64)]
    #[case::u64_1(1_u64, SEED, BITS, SHIFTS, 4035685362827872442_u64)]
    fn test_get_1d_noise<I, S, E>(#[case] input: I, #[case] seed: S, #[case] bits: impl IntoIterator<Item = BaseType>, #[case] shifts: impl IntoIterator<Item = u32>, #[case] expected: E)
    where
        I: AsPrimitive<BaseType>,
        S: AsPrimitive<BaseType>,
        E: AsPrimitive<BaseType> + PartialEq + Debug,
        BaseType: AsPrimitive<I>,
        BaseType: AsPrimitive<S>,
        BaseType: AsPrimitive<E>,
    {
        let bits = bits.into_iter().take(3);
        let shifts = shifts.into_iter().take(3);
        let result_a: E = get_1d_noise(input, seed, bits, shifts);
        let result_b = get_fast_1d_noise(input, seed);
        assert_eq!(result_a, result_b);
        assert_eq!(result_a, expected);
    }

    #[test]
    fn test_statistical_distribution() {
        const SAMPLE_SIZE: usize = 1_000_000;
        let mut rng = Rng::default();
        let mut samples = Vec::with_capacity(SAMPLE_SIZE);
        for _ in 0..SAMPLE_SIZE {
            samples.push(rng.rand());
        }
        let mean = (&samples).mean();
        let std_dev = (&samples).std_dev();
        assert!(
            // This is reasonably loose and related to the lower sample size
            mean >= 0.48 && mean <= 0.52 && std_dev >= 0.28 && std_dev <= 0.29,
            "Mean and Standard Deviation test failed:  mean={mean}, std_dev={std_dev}"
        );
    }

    #[test]
    fn test_identical_values() {
        let mut rng = Rng::default();

        let value = rng.rand();
        let a = rng.rand_from(value);
        let b = rng.rand_from(value);
        assert_eq!(a, b);
    }

    #[rstest]
    #[case::simple_number([0, 1, 2], Some(2))]
    #[case::simple_str(["a", "b", "c"], Some("c"))]
    #[case::simple_float([1.0, -1.0, 0.0], Some(-0.0))]
    #[case::more_floats(vec![1.0, -1.0, 0.0, 0.5], Some(0.5))]
    fn test_choose<'a, O>(#[case] data: impl IntoIterator<Item = O>, #[case] expected: Option<O>)
    where
        O: PartialEq + Debug + 'a,
    {
        let data = data.into_iter().collect::<Vec<_>>();
        let data_iter = data.iter();
        let seed: u32 = 0xDEADBEEF;
        let mut rng = Rng::from_seed(seed);
        let chosen = rng.choose(data_iter);
        assert_eq!(chosen, expected.as_ref());
    }

    #[rstest]
    #[case::u8_1d_0([0_u8], SEED, 115_u8)]
    #[case::u8_1d_1([1_u8], SEED, 57_u8)]
    #[case::u8_2d_0([0_u8, 1_u8], SEED, 104_u8)]
    #[case::u8_2d_1([1_u8, 0_u8], SEED, 192_u8)]
    #[case::u8_3d_0([0_u8, 1_u8, 2_u8], SEED, 163_u8)]
    #[case::u8_3d_1([1_u8, 0_u8, 2_u8], SEED, 251_u8)]
    #[case::u8_4d_0([0_u8, 1_u8, 2_u8, 3_u8], SEED, 53_u8)]
    #[case::u8_4d_1([1_u8, 0_u8, 2_u8, 3_u8], SEED, 141_u8)]
    #[case::u16_1d_0([0_u16], SEED, 43891_u16)]
    #[case::u16_1d_1([1_u16], SEED, 18233_u16)]
    #[case::u16_2d_0([0_u16, 1_u16], SEED, 3432_u16)]
    #[case::u16_2d_1([1_u16, 0_u16], SEED, 10432_u16)]
    #[case::u16_3d_0([0_u16, 1_u16, 2_u16], SEED, 53411_u16)]
    fn test_get_noise<I, S, E>(#[case] input: impl IntoIterator<Item = I>, #[case] seed: S, #[case] expected: E)
    where
        I: AsPrimitive<BaseType>,
        S: AsPrimitive<BaseType>,
        E: AsPrimitive<BaseType> + PartialEq + Debug + Sum<E> + WrappingAdd,
        BaseType: AsPrimitive<I>,
        BaseType: AsPrimitive<S>,
        BaseType: AsPrimitive<E>,
    {
        let result: E = get_noise(input.into_iter(), seed);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_rand_differs_with_seed() {
        let seed1 = SEED;
        let seed2 = 0xDEADBEEF_u64;
        let seed3 = 0xCAFEBABE_u64;

        let mut rng1 = Rng::new(seed1);
        let mut rng2 = Rng::new(seed2);
        let mut rng3 = Rng::new(seed3);

        let value1 = rng1.rand();
        let value2 = rng2.rand();
        let value3 = rng3.rand();

        assert_ne!(value1, value2);
        assert_ne!(value1, value3);
        assert_ne!(value2, value3);
    }

    #[rstest]
    #[case::vec(0..100)]
    fn test_gen_range_differs_with_seed<O>(#[case] data: impl IntoIterator<Item = O>)
    where
        O: Debug + PartialEq,
    {
        let seed1 = SEED;
        let seed2 = 0xDEADBEEF_u64;
        let seed3 = 0xCAFEBABE_u64;

        let mut rng1 = Rng::new(seed1);
        let mut rng2 = Rng::new(seed2);
        let mut rng3 = Rng::new(seed3);

        let data = data.into_iter().collect::<Vec<_>>();

        let value1 = rng1.choose(data.iter());
        let value2 = rng2.choose(data.iter());
        let value3 = rng3.choose(data.iter());

        assert_ne!(value1, value2);
        assert_ne!(value1, value3);
        assert_ne!(value2, value3);
    }
}
