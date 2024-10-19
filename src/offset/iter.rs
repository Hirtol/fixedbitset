use core::iter::{FusedIterator};
use core::marker::PhantomData;
use crate::generic::BitSet;
use crate::SimdBlock;

#[derive(Debug)]
pub struct OverlapIter<'a, I> {
    itr: I,
    overlap_len: usize,
    current: usize,
    _phantom: PhantomData<&'a ()>
}

pub fn new_overlap<'a>(self_bitset: &'a impl BitSet, other_bitset: &'a impl BitSet) -> OverlapIter<'a, impl Iterator<Item=(&'a SimdBlock, &'a SimdBlock)>> {
    let self_start = self_bitset.root_block_offset();
    let other_start = other_bitset.root_block_offset();

    let overlap_start = self_start.max(other_start);
    let overlap_end = self_bitset.root_block_len().min(other_bitset.root_block_len());

    let self_offset = overlap_start - self_start;
    let other_offset = overlap_start - other_start;
    let overlap_len = overlap_end.saturating_sub(overlap_start);
    // No need to `.take()` on `other_bitset` as our previous slice takes care of that.
    let itr = self_bitset.as_simd_blocks().skip(self_offset).take(overlap_len)
        .zip(other_bitset.as_simd_blocks().skip(other_offset));

    OverlapIter {
        itr,
        overlap_len,
        current: 0,
        _phantom: Default::default(),
    }
}

impl<'a, I: Iterator<Item=(&'a SimdBlock, &'a SimdBlock)>> Iterator for OverlapIter<'a, I> {
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

impl<'a, I: Iterator<Item=(&'a SimdBlock, &'a SimdBlock)>> ExactSizeIterator for OverlapIter<'a, I> {
    fn len(&self) -> usize {
        self.overlap_len - self.current
    }
}

impl<'a, I: FusedIterator<Item=(&'a SimdBlock, &'a SimdBlock)>> FusedIterator for OverlapIter<'a, I> {}