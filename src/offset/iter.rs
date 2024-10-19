use core::iter::FusedIterator;
use crate::offset::OffsetBitSet;
use crate::SimdBlock;

#[derive(Debug)]
pub struct OverlapIter<'a> {
    self_blocks: &'a [SimdBlock],
    other_blocks: &'a [SimdBlock],
    self_offset: usize,
    other_offset: usize,
    overlap_len: usize,
    current: usize,
}

impl<'a> OverlapIter<'a> {
    pub fn new(self_bitset: &'a OffsetBitSet<impl AsRef<[SimdBlock]>>, other_bitset: &'a OffsetBitSet<impl AsRef<[SimdBlock]>>) -> Option<Self> {
        let self_start = self_bitset.root_block_offset as usize;
        let self_len = self_bitset.len();
        let other_start = other_bitset.root_block_offset as usize;
        let other_len = other_bitset.len();

        let overlap_start = self_start.max(other_start);
        let overlap_end = (self_start + self_len).min(other_start + other_len);
        
        if overlap_start >= overlap_end {
            return None;
        }

        Some(Self {
            self_blocks: self_bitset.simd_blocks(),
            other_blocks: other_bitset.simd_blocks(),
            self_offset: overlap_start - self_start,
            other_offset: overlap_start - other_start,
            overlap_len: overlap_end - overlap_start,
            current: 0,
        })
    }
}

impl<'a> Iterator for OverlapIter<'a> {
    type Item = (&'a SimdBlock, &'a SimdBlock);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.overlap_len {
            let self_block = &self.self_blocks[self.self_offset + self.current];
            let other_block = &self.other_blocks[self.other_offset + self.current];
            self.current += 1;
            Some((self_block, other_block))
        } else {
            None
        }
    }
}

impl<'a> FusedIterator for OverlapIter<'a> {}