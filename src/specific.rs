use std::marker::PhantomData;
use crate::{FixedBitSet};
use crate::sparse::SparseBitSet;
use crate::SimdBlock;

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

impl<'a> SubBitSet<'a> for FixedBitSet {
    fn is_subset(&self, other: &FixedBitSet) -> bool {
        self.is_subset(other)
    }
}

impl<'a> SubBitSet<'a> for LazyAnd<'a, SparseBitSet<'a, &'a [SimdBlock]>, &FixedBitSet> {
    #[inline]
    fn is_subset(&self, other: &FixedBitSet) -> bool {
        let mut overlap = calculate_overlap(&self.left, self.right);
        overlap.all(|(overlap, new_itr)| {
            // SAFETY: Requires that `other` is at least as long as `self.right`.
            let other_itr = unsafe {
                other.as_simd_slice().get_unchecked(overlap.overlap_start..)
            };
            new_itr.map(|(x, y)| *x & *y)
                .zip(other_itr)
                // .all(|(x, y)| x.andnot(*y).is_empty())
                .all(|(x, y)| x & *y == x)
        })
    }
}

pub trait SubBitSet<'a>: Sized {
    fn to_sparse_set(self) -> SparseBitSet<'a, &'a [SimdBlock]> {
        panic!()
    }
    fn is_subset(&self, other: &FixedBitSet) -> bool;
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
            set.all(|(x, y)| {
                x.andnot(*y).is_empty()
            })
        });
        // Ensure that .all() wasn't empty!
        result && sets_handled != 0
    }
}

#[inline]
pub fn calculate_overlap<'a>(
    left: &'a SparseBitSet<'a, &'a [SimdBlock]>,
    right: &'a FixedBitSet,
) -> impl Iterator<Item = (crate::iter::OverlapState, impl Iterator<Item=(&'a SimdBlock, &'a SimdBlock)> + 'a)> {
    left.bit_sets().map(|sub_set| {
        let overlap = crate::iter::calculate_overlap(&sub_set, right);
        // SAFETY: The calculating in `calculate_overlap` ensures that nothing is ever out of bounds.
        let itr = unsafe {
            sub_set
                .blocks.get_unchecked(overlap.left_offset..overlap.left_offset + overlap.overlap_len)
                .iter()
                .zip(right.as_simd_slice().get_unchecked(overlap.right_offset..))
        };
        
        
        (overlap, itr)
    })
}

#[cfg(test)]
mod tests {
    use crate::FixedBitSet;
    // use crate::generic::BitSet;
    use super::SubBitSet;
    use crate::sparse::SparseBitSetCollection;

    #[test]
    pub fn test_lazy_and_large() {
        let mut fset = FixedBitSet::with_capacity(1000);
        let mut fset2 = FixedBitSet::with_capacity(1000);
        fset.insert_range(0..100);
        fset2.insert_range(400..600);
        let mut base_collection = SparseBitSetCollection::new();

        let left_idx = base_collection.push_collection(&[250, 450]);
        // let right_idx = base_collection.push_collection_itr((0..100).map(|i| i * 5));

        let left = base_collection.get_set_ref(left_idx);

        assert!(!left.is_subset(&fset));
        let combined = super::LazyAnd {
            left: left,
            right: &fset2,
            _phantom: Default::default(),
        };

        assert!(combined.is_subset(&fset2));
        assert!(!combined.is_subset(&fset));
        // let left_cont = left.as_simd_blocks().collect::<Vec<_>>();
        // let mut itr = left.as_simd_blocks();
        // println!("T: {:?}", itr.next_back());
        // while let Some(i) = itr.next() {
        //     println!("CONT: {i:?}");
        // }
        // let right = base_collection.get_set_ref(right_idx);

        // println!("ROOT: {:?}", base_collection);
        // println!("CONTENT: {:?}", left_cont);
        // let things = left.ones().collect::<Vec<_>>();
        // let combined = left.lazy_and(&fset2);
        // let combined2 = combined.ones().collect::<Vec<_>>();
        // let comb_data = combined.as_simd_blocks().collect::<Vec<_>>();
        // println!("STUFF: {combined2:?}\n{things:?}");
        // println!("COMB: {:?}", comb_data);
        // assert!(!combined.is_subset(&fset));
    }
}