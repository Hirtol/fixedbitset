use core::marker::PhantomData;
use crate::{Block, FixedBitSet, Ones, SimdBlock, Zeroes, BITS};
use crate::iter::SimdToSubIter;
use crate::offset::iter::OverlapIter;

pub trait BitSet: Sized {
    /// Return the internal SIMD blocks of the [BitSet]
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item=SimdBlock> + DoubleEndedIterator;

    /// Return sub-block representations of the [SimdBlock]s
    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item=Block> + DoubleEndedIterator;

    /// Return the amount of bits allocated to this [BitSet]
    fn bit_len(&self) -> usize;

    /// Return the amount of [SimdBlock]s which should be skipped when comparing with a 'root' [FixedBitSet].
    fn root_block_offset(&self) -> usize;

    /// Check whether `self ⊆ other`, aka, if all bits present in `self` are present in `other`
    #[inline]
    fn is_subset(&self, other: &impl BitSet) -> bool {
        // Can't possibly be a subset if the root index is greater!
        if self.root_block_len() > other.root_block_len() {
            return false;
        }
        let mut overlap = self.overlap(other);
        // Technically the empty set is a subset of all other sets, but we ignore that here >.>
        if overlap.len() == 0 {
            false
        } else {
            overlap.all(|(x, y)| x.andnot(y).is_empty())
        }
    }

    /// Returns `true` if the set is a superset of another, i.e. `self` contains
    /// at least all the values in `other`.
    #[inline(always)]
    fn is_superset(&self, other: &impl BitSet) -> bool {
        other.is_subset(self)
    }

    /// Yield all [SimdBlock]s which overlap between the two given sets
    #[inline]
    fn overlap<'a>(
        &'a self,
        other: &'a impl BitSet,
    ) -> OverlapIter<'a, impl DoubleEndedIterator<Item=(SimdBlock, SimdBlock)> + ExactSizeIterator> {
        crate::iter::new_overlap_simd(self, other)
    }

    /// An efficient way of checking whether `self` and `other` have any overlapping [SimdBlock]s
    #[inline(always)]
    fn has_overlap(&self, other: &impl BitSet) -> bool {
        let self_start = self.root_block_offset();
        let other_start = other.root_block_offset();

        // Check if there is an overlap between the two ranges
        !(self.root_block_len() <= other_start || other.root_block_len() <= self_start)
    }

    #[inline]
    fn ones(&self) -> impl Iterator<Item = usize> {
        let offset = self.root_block_offset() * SimdBlock::BITS;
        ones_impl(self).map(move |i| i + offset)
    }

    /// Iterates over all disabled bits.
    ///
    /// Iterator element is the index of the `0` bit, type `usize`.
    #[inline]
    fn zeroes(&self) -> impl Iterator<Item = usize> {
        let offset = self.root_block_offset() * SimdBlock::BITS;
        zeroes_impl(self).map(move |i| i + offset)
    }

    /// Return the length of the OffsetBitSet if it was instead a full [FixedBitSet]
    #[inline(always)]
    fn root_block_len(&self) -> usize {
        self.root_block_offset() + self.as_simd_blocks().len()
    }

    /// Create a new lazy bitset which will only perform the `AND` when needed.
    ///
    /// Note that the result is not cached, so it would be repeated every invocation!
    #[inline(always)]
    fn lazy_and<T: BitSet>(&self, other: T) -> LazyAnd<&Self, T> {
        LazyAnd {
            left: self,
            right: other,
            _phantom: Default::default(),
        }
    }

    /// Create a new lazy bitset which will only perform the `AND` when needed.
    ///
    /// Note that the result is not cached, so it would be repeated every invocation!
    #[inline(always)]
    fn to_lazy_and<'a, T: BitSet>(self, other: T) -> LazyAnd<'a, Self, T> {
        LazyAnd {
            left: self,
            right: other,
            _phantom: Default::default(),
        }
    }

    /// Turn this [BitSet] into a [FixedBitSet] with the given capacity.
    ///
    /// # Panic
    ///
    /// Will panic if the number of bits is too small too hold the values of the current set.
    fn as_fixed_bit_set(&self, bits: usize) -> FixedBitSet {
        let bits_to_start = self.root_block_offset() * SimdBlock::BITS;
        let total_bits = bits_to_start + self.as_simd_blocks().len() * SimdBlock::BITS;
        assert!(total_bits < bits, "Creating a FixedBitSet out of an OffsetBitSet requires the total `bits` ({bits}) count to be larger than the OffsetBitSet's size ({total_bits})");

        let sblock_count = self.root_block_offset() * SimdBlock::USIZE_COUNT;
        let repeat = core::iter::repeat_n(0, sblock_count)
            .chain(self.as_simd_blocks().flat_map(|v| v.into_usize_array()))
            .chain(core::iter::repeat(0));
        FixedBitSet::with_capacity_and_blocks(bits, repeat)
    }
}

impl<'a, T: BitSet> BitSet for &'a T {
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item=SimdBlock> + DoubleEndedIterator {
        (**self).as_simd_blocks()
    }

    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item=Block> + DoubleEndedIterator {
        (**self).as_sub_blocks()
    }

    fn bit_len(&self) -> usize {
        (**self).bit_len()
    }

    fn root_block_offset(&self) -> usize {
        (**self).root_block_offset()
    }
}

pub struct LazyAnd<'a, A, B> {
    left: A,
    right: B,
    _phantom: PhantomData<&'a ()>
}

impl<'a, A: BitSet, B: BitSet> BitSet for LazyAnd<'a, A, B> {
    #[inline]
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item=SimdBlock> + DoubleEndedIterator {
        self.left.overlap(&self.right)
            .map(|(x, y)| x & y)
    }

    #[inline(always)]
    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item=Block> + DoubleEndedIterator {
        crate::iter::new_overlap_sub_blocks(&self.left, &self.right)
            .map(|(x, y)| x & y)
    }

    #[inline(always)]
    fn bit_len(&self) -> usize {
        self.left.bit_len().min(self.right.bit_len())
    }

    #[inline(always)]
    fn root_block_offset(&self) -> usize {
        crate::iter::overlap_start(&self.left, &self.right)
    }
}

#[inline]
pub(crate) fn ones_impl<'a>(set: &'a impl BitSet) -> Ones<'a, impl ExactSizeIterator<Item=usize> + DoubleEndedIterator + 'a> {
    let mut itr = set.as_sub_blocks();
    if let Some(first_block) = itr.next() {
        let last_block = itr.next_back().unwrap_or(0);
        Ones {
            bitset_front: first_block,
            bitset_back: last_block,
            block_idx_front: 0,
            block_idx_back: (1 + itr.len()) * BITS,
            remaining_blocks: itr,
            _phantom: Default::default(),
        }
    } else {
        Ones {
            bitset_front: 0,
            bitset_back: 0,
            block_idx_front: 0,
            block_idx_back: 0,
            remaining_blocks: itr,
            _phantom: Default::default(),
        }
    }
}

#[inline]
pub(crate) fn zeroes_impl<'a>(set: &'a impl BitSet) -> Zeroes<'a, impl ExactSizeIterator<Item=usize> + DoubleEndedIterator + 'a> {
    let mut itr = set.as_sub_blocks();
    match itr.next() {
        Some(block) => Zeroes {
            bitset: !block,
            block_idx: 0,
            len: set.bit_len(),
            remaining_blocks: itr,
            _phantom: Default::default(),
        },
        None => Zeroes {
            bitset: !0,
            block_idx: 0,
            len: set.bit_len(),
            remaining_blocks: itr,
            _phantom: Default::default(),
        },
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use crate::generic::{BitSet, LazyAnd};
    use crate::OffsetBitSetCollection;

    #[test]
    pub fn test_lazy_and() {
        let mut base_collection = OffsetBitSetCollection::new();
        
        let left_idx = base_collection.push_collection(&[128, 129, 256]);
        let right_idx = base_collection.push_collection(&[129, 256]);
        
        let left = base_collection.get_set_ref(left_idx);
        let right = base_collection.get_set_ref(right_idx);
        
        let combined = LazyAnd {
            left: &left,
            right: &right,
            _phantom: Default::default(),
        };

        let out = combined.ones().collect::<Vec<_>>();
        assert_eq!(out, vec![129, 256]);
    }
}