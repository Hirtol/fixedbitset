use crate::{Block, FixedBitSet};
use crate::SimdBlock;
use alloc::vec::Vec;

pub struct OffsetBitSetCollection {
    /// Tracks the offset in [SimdBlock] counts into the `block` array.
    pub offsets: Vec<u32>,
    pub blocks: Vec<SimdBlock>
}

impl OffsetBitSetCollection {
    pub fn new() -> Self {
        Self {
            offsets: alloc::vec![0],
            blocks: Vec::new(),
        }
    }

    /// Retrieve the given `bit_set` from storage
    #[inline]
    pub fn get_set_ref(&self, bit_set: usize) -> OffsetBitSetRef<'_> {
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_ref_unchecked(bit_set)
        }
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_ref_unchecked(&self, bit_set: usize) -> OffsetBitSetRef<'_> {
        let offset = *self.offsets.get_unchecked(bit_set);
        let next_offset = *self.offsets.get_unchecked(bit_set + 1);
        OffsetBitSetRef {
            offset,
            blocks: self.blocks.get_unchecked(offset as usize..next_offset as usize),
        }
    }
    
    #[inline]
    pub fn get_set_as_slice(&self, bit_set: usize) -> &[SimdBlock] {
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_unchecked(bit_set)
        }
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_unchecked(&self, bit_set: usize) -> &[SimdBlock] {
        let offset = *self.offsets.get_unchecked(bit_set);
        let next_offset = *self.offsets.get_unchecked(bit_set + 1);
        self.blocks.get_unchecked(offset as usize..next_offset as usize)
    }


    /// Return the length of the current collection
    #[inline]
    pub fn len(&self) -> usize {
        // We'll always have the last offset be there to mark the end of the `blocks` array.
        self.offsets.len() - 1
    }
}

pub struct OffsetBitSetRef<'a> {
    offset: u32,
    blocks: &'a [crate::SimdBlock]
}

impl<'a> OffsetBitSetRef<'a> {
    #[inline(always)]
    pub fn from_fixed_set(offset_start: u32, offset_end: u32, other: &'a FixedBitSet) -> Self {
        Self {
            offset: offset_start,
            blocks: &other.as_simd_slice()[offset_start as usize..offset_end as usize],
        }
    }

    /// Tu
    pub fn as_fixed_bit_set(&self, bits: usize) -> FixedBitSet {
        let bits_to_start = self.offset as usize * SimdBlock::BITS;
        let total_bits = bits_to_start + self.blocks.len() * SimdBlock::BITS;
        assert!(total_bits < bits, "Creating a FixedBitSet out of an OffsetBitSet requires the total `bits` ({bits}) count to be larger than the OffsetBitSet's size ({total_bits})");
        
        let repeat = core::iter::repeat_n(0, self.offset as usize)
            .chain(self.blocks.iter().flat_map(|v| v.into_usize_array()))
            .chain(core::iter::repeat(0));
        FixedBitSet::with_capacity_and_blocks(bits, repeat)
    }
}