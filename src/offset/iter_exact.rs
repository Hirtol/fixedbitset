use crate::generic::BitSet;
use crate::{Block, SimdBlock};
use core::iter::FusedIterator;
use core::marker::PhantomData;

#[derive(Debug)]
pub struct OverlapState {
    pub left_offset: usize,
    pub right_offset: usize,
    pub overlap_start: usize,
    pub overlap_len: usize,
}

#[inline]
pub fn calculate_overlaps<'a>(
    left: &'a impl BitSet,
    right: &'a impl BitSet,
) -> impl ExactSizeIterator<Item = OverlapState> + DoubleEndedIterator + 'a {
    left.root_block_offsets()
        .zip(right.root_block_offsets())
        .map(|(self_start, other_start)| {
            let overlap_start = self_start.max(other_start);
            let overlap_end = left.root_block_len().min(right.root_block_len());

            let left_offset = overlap_start - self_start;
            let right_offset = overlap_start - other_start;
            let overlap_len = overlap_end.saturating_sub(overlap_start);

            OverlapState {
                left_offset,
                right_offset,
                overlap_start,
                overlap_len,
            }
        })
}

pub struct ExactOverlapItr<'a, I> {
    itr: I,
    total_len: usize,
    _phantom: PhantomData<&'a ()>,
}

#[derive(Debug)]
pub struct NewOverlapIter<'a, I> {
    itr: I,
    overlap: OverlapState,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, T, I: Iterator<Item = (T, T)>> Iterator for NewOverlapIter<'a, I> {
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

impl<'a, T, I: ExactSizeIterator<Item = (T, T)>> ExactSizeIterator for NewOverlapIter<'a, I> {
    #[inline]
    fn len(&self) -> usize {
        self.itr.len()
    }
}

impl<'a, T, I> DoubleEndedIterator for NewOverlapIter<'a, I>
where
    I: DoubleEndedIterator + Iterator<Item = (T, T)>,
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

pub fn new_overlap_sub_simd<'a>(
    left: &'a impl BitSet,
    right: &'a impl BitSet,
) -> ExactOverlapItr<
    'a,
    impl DoubleEndedIterator<Item = NewOverlapIter<'a, impl DoubleEndedIterator<Item = (SimdBlock, SimdBlock)> + ExactSizeIterator + 'a>> + ExactSizeIterator + 'a,
> {
    let overlaps = calculate_overlaps(left, right);
    let total_len = calculate_overlaps(left, right).map(|ov| ov.overlap_len).sum();
    
    let itr = overlaps.map(|overlap| {
        NewOverlapIter {
            itr: left.as_simd_blocks()
                .skip(overlap.left_offset)
                .take(overlap.overlap_len)
                .zip(right.as_simd_blocks().skip(overlap.right_offset)),
            overlap,
            _phantom: Default::default(),
        }
    });

    ExactOverlapItr {
        itr,
        total_len,
        _phantom: Default::default(),
    }
}

impl<'a, T, I: Iterator<Item = NewOverlapIter<'a, J>>, J: Iterator<Item = (T, T)>> Iterator for ExactOverlapItr<'a, I> {
    type Item = NewOverlapIter<'a, J>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_itr) = self.itr.next() {
            self.total_len -= next_itr.overlap.overlap_len;
            Some(next_itr)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_len, Some(self.total_len))
    }
}

impl<'a, T, I: Iterator<Item = NewOverlapIter<'a, J>>, J: Iterator<Item = (T, T)>> ExactSizeIterator for ExactOverlapItr<'a, I>
{
    #[inline]
    fn len(&self) -> usize {
        self.total_len
    }
}

impl<'a, T, I, J: Iterator<Item = (T, T)>> DoubleEndedIterator for ExactOverlapItr<'a, I>
where
    I: DoubleEndedIterator + Iterator<Item = NewOverlapIter<'a, J>>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.itr.next_back() {
            self.total_len -= item.overlap.overlap_len;
            Some(item)
        } else {
            None
        }
    }
}