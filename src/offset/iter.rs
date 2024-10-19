use core::iter::{FusedIterator, Zip};
use core::slice::Iter;
use crate::generic::BitSet;
use crate::SimdBlock;

#[derive(Debug)]
pub struct OverlapIter<'a> {
    itr: Zip<Iter<'a, SimdBlock>, Iter<'a, SimdBlock>>,
    overlap_len: usize,
    current: usize,
}

impl<'a> OverlapIter<'a> {
    pub fn new(self_bitset: &'a impl BitSet, other_bitset: &'a impl BitSet) -> Self {
        let self_start = self_bitset.root_block_offset();
        let other_start = other_bitset.root_block_offset();

        let overlap_start = self_start.max(other_start);
        let overlap_end = self_bitset.root_block_len().min(other_bitset.root_block_len());
        
        let self_offset = overlap_start - self_start;
        let other_offset = overlap_start - other_start;
        let overlap_len = overlap_end.saturating_sub(overlap_start);
        
        let itr = self_bitset.as_simd_blocks()[self_offset..self_offset + overlap_len]
            .iter()
            .zip(&other_bitset.as_simd_blocks()[other_offset..other_offset + overlap_len]);
        
        Self {
            itr,
            overlap_len,
            current: 0,
        }
    }
}

impl<'a> Iterator for OverlapIter<'a> {
    type Item = (&'a SimdBlock, &'a SimdBlock);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.current += 1;
        self.itr.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.overlap_len - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for OverlapIter<'a> {
    fn len(&self) -> usize {
        self.overlap_len - self.current
    }
}

impl<'a> FusedIterator for OverlapIter<'a> {}