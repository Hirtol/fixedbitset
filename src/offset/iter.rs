use core::iter::{FusedIterator};
use core::marker::PhantomData;
use crate::generic::BitSet;
use crate::{Block, SimdBlock};

#[derive(Debug)]
pub struct OverlapIter<'a, I> {
    itr: I,
    _phantom: PhantomData<&'a ()>
}

pub fn new_overlap_simd<'a>(self_bitset: &'a impl BitSet, other_bitset: &'a impl BitSet) -> OverlapIter<'a, impl DoubleEndedIterator<Item=(SimdBlock, SimdBlock)> + ExactSizeIterator + 'a> {
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
        _phantom: Default::default(),
    }
}

pub fn new_overlap_sub_blocks<'a>(self_bitset: &'a impl BitSet, other_bitset: &'a impl BitSet) -> OverlapIter<'a, impl DoubleEndedIterator<Item=(Block, Block)> + ExactSizeIterator + 'a> {
    let self_start = self_bitset.root_block_offset() * SimdBlock::USIZE_COUNT;
    let other_start = other_bitset.root_block_offset() * SimdBlock::USIZE_COUNT;

    let overlap_start = self_start.max(other_start);
    let overlap_end = self_bitset.root_block_len().min(other_bitset.root_block_len()) * SimdBlock::USIZE_COUNT;

    let self_offset = overlap_start - self_start;
    let other_offset = overlap_start - other_start;
    let overlap_len = overlap_end.saturating_sub(overlap_start);
    // No need to `.take()` on `other_bitset` as our previous slice takes care of that.
    let itr = self_bitset.as_sub_blocks().skip(self_offset).take(overlap_len)
        .zip(other_bitset.as_sub_blocks().skip(other_offset));

    OverlapIter {
        itr,
        _phantom: Default::default(),
    }
}

#[inline]
pub fn overlap_start(left: &impl BitSet, right: &impl BitSet) -> usize {
    let self_start = left.root_block_offset();
    let other_start = right.root_block_offset();

    self_start.max(other_start)
}

impl<'a, T, I: Iterator<Item=(T, T)>> Iterator for OverlapIter<'a, I> {
    type Item = (T, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.itr.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.itr.size_hint()
    }
}

impl<'a, T, I: ExactSizeIterator<Item=(T, T)>> ExactSizeIterator for OverlapIter<'a, I> {
    #[inline]
    fn len(&self) -> usize {
        self.itr.len()
    }
}

impl<'a, T, I> DoubleEndedIterator for OverlapIter<'a, I>
where
    I: DoubleEndedIterator + Iterator<Item=(T, T)>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.itr.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.itr.nth_back(n)
    }
}

impl<'a, T, I: FusedIterator<Item=(T, T)>> FusedIterator for OverlapIter<'a, I> {}

pub struct SimdToSubIter<I> {
    current_idx: usize,
    current_block: Option<[usize; SimdBlock::USIZE_COUNT]>,
    last_block: Option<[usize; SimdBlock::USIZE_COUNT]>,
    last_idx: usize,
    itr: I
}

impl<I: Iterator<Item=SimdBlock> + DoubleEndedIterator> SimdToSubIter<I> {
    pub fn new(mut blocks: I) -> Self {
        let block = blocks.next();
        let last_block = blocks.next_back();
        Self {
            current_idx: 0,
            current_block: block.map(|i| i.into_usize_array()),
            last_block: last_block.map(|i| i.into_usize_array()),
            last_idx: SimdBlock::USIZE_COUNT - 1,
            itr: blocks,
        }
    }
}

impl<I: Iterator<Item=SimdBlock>> Iterator for SimdToSubIter<I> {
    type Item = Block;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.current_block, self.last_block) {
            (Some(block), _) => {
                let out = block[self.current_idx];
                self.current_idx += 1;

                if self.current_idx >= SimdBlock::USIZE_COUNT {
                    self.current_block = self.itr.next().map(|i| i.into_usize_array());
                    self.current_idx = 0;
                }

                Some(out)
            },
            (_, Some(block)) => {
                let out = block[(SimdBlock::USIZE_COUNT - 1) - self.last_idx];
                let (next_value, overflow) = self.last_idx.overflowing_sub(1);
                if overflow {
                    // We know that there is nothing else, otherwise `current_block` would've been true
                    self.last_block = None;
                    self.last_idx = SimdBlock::USIZE_COUNT - 1;
                } else {
                    self.last_idx = next_value;
                }
                
                Some(out)
            },
            (None, None) => {
                None
            },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (left, right) = self.itr.size_hint();
        
        (left * SimdBlock::USIZE_COUNT, right.map(|i| i * SimdBlock::USIZE_COUNT))
    }
}

impl<I: DoubleEndedIterator<Item=SimdBlock>> DoubleEndedIterator for SimdToSubIter<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match (self.current_block, self.last_block) {
            (_, Some(block)) => {
                let out = block[self.last_idx];

                let (next_value, overflow) = self.last_idx.overflowing_sub(1);
                if overflow {
                    self.last_block = self.itr.next().map(|i| i.into_usize_array());
                    self.last_idx = SimdBlock::USIZE_COUNT - 1;
                } else {
                    self.last_idx = next_value;
                }

                Some(out)
            },
            (Some(block), _) => {
                let out = block[SimdBlock::USIZE_COUNT - 1 - self.current_idx];
                self.current_idx += 1;

                if self.current_idx >= SimdBlock::USIZE_COUNT {
                    self.current_block = None;
                    self.current_idx = 0;
                }

                Some(out)
            },
            (None, None) => {
                None
            },
        }
    }
}

impl<I: ExactSizeIterator<Item=SimdBlock>> ExactSizeIterator for SimdToSubIter<I> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.itr.len() * SimdBlock::USIZE_COUNT
    }
}