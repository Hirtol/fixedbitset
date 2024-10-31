use crate::iter::{OverlapState, SimdToSubIter};
use crate::sparse::{SparseBitSet, SparseBitSetRef};
use crate::FixedBitSet;
use crate::SimdBlock;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct LazyAnd<'a, A, B> {
    left: A,
    right: B,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, A, B> LazyAnd<'a, A, B> {
    pub fn new(left: A, right: B) -> Self {
        LazyAnd {
            left,
            right,
            _phantom: Default::default(),
        }
    }
}

impl<'a> LazyAnd<'a, SparseBitSet<'a, &'a [SimdBlock]>, &FixedBitSet> {
    pub fn ones(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.anded().flat_map(|(overlap, itr)| {
            let itr = SimdToSubIter::new(itr);
            let bit_offset = overlap.overlap_start * SimdBlock::BITS;

            crate::generic::ones_impl(itr).map(move |i| bit_offset + i)
        })
    }

    pub fn anded(
        &self,
    ) -> impl DoubleEndedIterator<
        Item = (
            OverlapState,
            impl DoubleEndedIterator<Item = SimdBlock> + ExactSizeIterator + '_,
        ),
    > + ExactSizeIterator {
        calculate_overlap(&self.left, self.right)
            .map(|(overlap, new_itr)| (overlap, new_itr.map(|(x, y)| *x & *y)))
    }
}

impl<'a> SubBitSet<'a> for FixedBitSet {
    fn is_subset(&self, other: &FixedBitSet) -> bool {
        self.is_subset(other)
    }

    fn count_ones(&self) -> usize {
        self.count_ones(..)
    }
}

impl FixedBitSet {
    /// Compute the bitwise OR of this fixed bitset with the given sparse set.
    pub fn sparse_union_with(&mut self, other: &SparseBitSetRef<'_>) {
        other.bit_sets().for_each(|sub_set| {
            let overlap = crate::iter::calculate_overlap(&sub_set, self);
            // SAFETY: The calculating in `calculate_overlap` ensures that nothing is ever out of bounds.
            unsafe {
                sub_set
                    .blocks
                    .get_unchecked(overlap.left_offset..overlap.left_offset + overlap.overlap_len)
                    .iter()
                    .zip(self.as_mut_simd_slice().get_unchecked_mut(overlap.right_offset..))
                    .for_each(|(x, y)| *y |= *x)
            };
        });
    }
}

impl<'a> SubBitSet<'a> for LazyAnd<'a, SparseBitSet<'a, &'a [SimdBlock]>, &FixedBitSet> {
    #[inline]
    fn is_subset(&self, other: &FixedBitSet) -> bool {
        self.anded().all(|(overlap, new_itr)| {
            // SAFETY: Requires that `other` is at least as long as `self.right`.
            let other_itr = unsafe { other.as_simd_slice().get_unchecked(overlap.overlap_start..) };
            new_itr.zip(other_itr).all(|(x, y)| x.andnot(*y).is_empty())
        })
    }

    fn count_ones(&self) -> usize {
        self.anded()
            .map(|(_, new_itr)| {
                new_itr.map(|block| {
                    block
                        .into_usize_array()
                        .iter()
                        .map(|sub| sub.count_ones() as usize)
                        .sum::<usize>()
                }).sum::<usize>()
            })
            .sum()
    }
}

pub trait SubBitSet<'a>: Sized {
    fn to_sparse_set(self) -> SparseBitSet<'a, &'a [SimdBlock]> {
        panic!()
    }
    
    fn is_subset(&self, other: &FixedBitSet) -> bool;

    /// Count the amount of ones available in this full bitset.
    fn count_ones(&self) -> usize;
}

impl<'a> SubBitSet<'a> for SparseBitSet<'a, &'a [SimdBlock]> {
    #[inline(always)]
    fn to_sparse_set(self) -> SparseBitSet<'a, &'a [SimdBlock]> {
        self
    }

    #[inline]
    fn is_subset(&self, other: &FixedBitSet) -> bool {
        let mut overlap = calculate_overlap(self, other);
        let mut sets_handled = 0;

        let result = overlap.all(|(_, mut set)| {
            sets_handled += 1;
            set.all(|(x, y)| x.andnot(*y).is_empty())
        });
        // Ensure that .all() wasn't empty!
        result && sets_handled != 0
    }

    fn count_ones(&self) -> usize {
        self.sub_blocks()
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum()
    }
}

#[inline]
pub fn calculate_overlap<'a>(
    left: &'a SparseBitSet<'a, &'a [SimdBlock]>,
    right: &'a FixedBitSet,
) -> impl DoubleEndedIterator<
    Item = (
        crate::iter::OverlapState,
        impl DoubleEndedIterator<Item = (&'a SimdBlock, &'a SimdBlock)> + ExactSizeIterator + 'a,
    ),
> + ExactSizeIterator {
    left.bit_sets().map(|sub_set| {
        let overlap = crate::iter::calculate_overlap(&sub_set, right);
        // SAFETY: The calculating in `calculate_overlap` ensures that nothing is ever out of bounds.
        let itr = unsafe {
            sub_set
                .blocks
                .get_unchecked(overlap.left_offset..overlap.left_offset + overlap.overlap_len)
                .iter()
                .zip(right.as_simd_slice().get_unchecked(overlap.right_offset..))
        };

        (overlap, itr)
    })
}

#[cfg(test)]
mod tests {
    use super::SubBitSet;
    use crate::sparse::SparseBitSetCollection;
    use crate::FixedBitSet;
    use crate::iter::SimdToSubIter;

    #[test]
    pub fn test_lazy_and_large() {
        let mut fset = FixedBitSet::with_capacity(1000);
        let mut fset2 = FixedBitSet::with_capacity(1000);
        fset.insert_range(0..100);
        fset2.insert_range(400..999);
        let mut base_collection = SparseBitSetCollection::new();

        let left_idx = base_collection.push_collection(&[453, 454, 456, 457, 480, 489, 492, 494, 495, 497, 498, 500, 503, 504, 506, 595]);

        let left = base_collection.get_set_ref(left_idx);

        assert!(!left.is_subset(&fset));
        let combined = super::LazyAnd {
            left,
            right: &fset2,
            _phantom: Default::default(),
        };

        assert!(combined.is_subset(&fset2));
        assert!(!combined.is_subset(&fset));
        let data= combined.anded().flat_map(|i| i.1).collect::<Vec<_>>();
        let other_data = combined.anded().flat_map(|i| SimdToSubIter::new(i.1)).collect::<Vec<_>>();
        println!("Data: {data:?}\nOther: {other_data:?}");
        let values = combined.ones().collect::<Vec<_>>();
        println!("VALUES: {values:?} - {combined:?}");
    }

    #[test]
    pub fn test_sparse_union() {
        let mut fset = FixedBitSet::with_capacity(1000);
        let mut fset2 = FixedBitSet::with_capacity(1000);
        fset.insert_range(0..100);
        fset2.insert_range(400..600);
        
        let mut base_collection = SparseBitSetCollection::new();

        let left_idx = base_collection.push_collection(&[250, 450]);

        let left = base_collection.get_set_ref(left_idx);
        fset.sparse_union_with(&left);
        
        assert!(fset.contains(250));
        assert!(fset.contains(450));
        assert!(!fset.contains(500));
    }

    #[test]
    pub fn test_count_ones() {
        let mut base_collection = SparseBitSetCollection::new();

        let right_idx = base_collection.push_collection_itr((400..900));
        let right = base_collection.get_set_ref(right_idx);
        assert_eq!(right.count_ones(), 500);
        
        let left_idx = base_collection.push_collection(&[250, 450]);
        let left = base_collection.get_set_ref(left_idx);
        assert_eq!(left.count_ones(), 2);

        let mut fset = FixedBitSet::with_capacity(1000);
        fset.insert_range(400..600);
        
        let combined = super::LazyAnd {
            left,
            right: &fset,
            _phantom: Default::default(),
        };
        assert_eq!(combined.count_ones(), 1);
    }
}
