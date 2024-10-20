#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Block(pub __m256d);

impl Block {
    #[inline]
    pub fn is_empty(self) -> bool {
        unsafe {
            let value = _mm256_castpd_si256(self.0);
            _mm256_testz_si256(value, value) == 1
        }
    }

    #[inline]
    pub fn andnot(self, other: Self) -> Self {
        unsafe { Self(_mm256_andnot_pd(other.0, self.0)) }
    }
}

impl Not for Block {
    type Output = Block;
    #[inline]
    fn not(self) -> Self::Output {
        unsafe { Self(_mm256_xor_pd(self.0, Self::ALL.0)) }
    }
}

impl BitAnd for Block {
    type Output = Block;
    #[inline]
    fn bitand(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_and_pd(self.0, other.0)) }
    }
}

impl BitAndAssign for Block {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm256_and_pd(self.0, other.0);
        }
    }
}

impl BitOr for Block {
    type Output = Block;
    #[inline]
    fn bitor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_or_pd(self.0, other.0)) }
    }
}

impl BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm256_or_pd(self.0, other.0);
        }
    }
}

impl BitXor for Block {
    type Output = Block;
    #[inline]
    fn bitxor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_xor_pd(self.0, other.0)) }
    }
}

impl BitXorAssign for Block {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        unsafe { self.0 = _mm256_xor_pd(self.0, other.0) }
    }
}

impl PartialEq for Block {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let new = _mm256_xor_pd(self.0, other.0);
            let neq = _mm256_castpd_si256(new);
            _mm256_testz_si256(neq, neq) == 1
        }
    }
}
