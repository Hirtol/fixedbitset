use crate::{Block, Ones, SimdBlock, Zeroes, BITS};
use crate::offset::iter::OverlapIter;

pub trait BitSet: Sized {
    /// Return the internal SIMD blocks of the [BitSet]
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item=&SimdBlock> + DoubleEndedIterator;

    /// Return sub-block representations of the [SimdBlock]s
    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item=&Block> + DoubleEndedIterator;

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
            overlap.all(|(x, y)| x.andnot(*y).is_empty())
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
    ) -> OverlapIter<'a, impl Iterator<Item=(&'a SimdBlock, &'a SimdBlock)>> {
        crate::iter::new_overlap(self, other)
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
}

pub struct LazyApplication {

}

#[inline]
pub(crate) fn ones_impl(set: &impl BitSet) -> Ones<impl ExactSizeIterator<Item=&usize>> {
    let mut itr = set.as_sub_blocks();
    if let Some(&first_block) = itr.next() {
        let last_block = *itr.next_back().unwrap_or(&0);
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
pub(crate) fn zeroes_impl(set: &impl BitSet) -> Zeroes {
    // match set.as_sub_blocks().split_first() {
    //     Some((&block, rem)) => Zeroes {
    //         bitset: !block,
    //         block_idx: 0,
    //         len: set.bit_len(),
    //         remaining_blocks: rem.iter(),
    //     },
    //     None => Zeroes {
    //         bitset: !0,
    //         block_idx: 0,
    //         len: set.bit_len(),
    //         remaining_blocks: [].iter(),
    //     },
    // }
    todo!()
}