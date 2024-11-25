use crate::generic::BitSet;
use crate::iter::SimdToSubIter;
use crate::{Ones, SimdBlock, BITS};
use crate::{Block, OffsetBitSetRef};
use alloc::vec::Vec;
use core::iter::FlatMap;

type SubOffsetIdx = u32;
// pub type SparseBitSetOwned = SparseBitSet<'a, Vec<SimdBlock>>;
pub type SparseBitSetRef<'a> = SparseBitSet<'a, &'a [SimdBlock]>;
pub type SparseBitSetMut<'a> = SparseBitSet<'a, &'a mut [SimdBlock]>;

#[derive(Debug, Clone)]
pub struct SparseBitSetCollection {
    /// Tracks the offset in [SimdBlock] counts into the `block` array.
    ///
    /// # Invariant
    /// There is always a last [SubOffsetIdx] to mark the end of the array
    offsets: Vec<SubOffsetIdx>,
    sub_offsets: Vec<PseudoOffset>,
    blocks: Vec<SimdBlock>,
}

#[derive(Default, Debug, Clone, Copy)]
struct PseudoOffset {
    blocks_offset: u32,
    root_bitset_offset: u32,
}

impl Default for SparseBitSetCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseBitSetCollection {
    pub fn new() -> Self {
        Self {
            offsets: alloc::vec![0],
            sub_offsets: alloc::vec![PseudoOffset {
                blocks_offset: 0,
                root_bitset_offset: 0,
            }],
            blocks: Vec::new(),
        }
    }

    /// Push a _sorted_ collection of bits which are expected to be enabled as a new [crate::OffsetBitSet].
    ///
    /// A base offset will automatically be inferred based on the first item in the list
    ///
    /// # Returns
    /// The index to access the new [crate::OffsetBitSet]
    pub fn push_collection(&mut self, bits: &[usize]) -> usize {
        self.push_collection_itr(bits.iter().copied())
    }

    /// Push a _sorted_ collection of bits which are expected to be enabled as a new [SparseBitSet].
    ///
    /// A base offset will automatically be inferred based on the first item in the list
    ///
    /// # Returns
    /// The index to access the new [SparseBitSet]
    pub fn push_collection_itr(&mut self, mut bits: impl Iterator<Item = usize>) -> usize {
        let Some(first_item) = bits.next() else {
            panic!("Need a non-empty set");
        };
        
        let root_block_offset = first_item / SimdBlock::BITS;
        let mut pseudo = PseudoOffset {
            blocks_offset: self.blocks.len() as u32,
            root_bitset_offset: root_block_offset as u32,
        };
        let new_entry = self
            .sub_offsets
            .last_mut()
            .expect("Impossible invariant violation");
        *new_entry = pseudo;
        let mut current_block_index = root_block_offset;
        let mut current_block = [0; SimdBlock::USIZE_COUNT];

        for bit in core::iter::once(first_item).chain(bits) {
            let (sub_block, remaining) = crate::div_rem(bit, crate::BITS);
            let block_index = sub_block / SimdBlock::USIZE_COUNT;

            // Fill the space between this block and the next, can result in a lot of empty spots for discontiguous sets, but that's the trade-off
            if block_index > current_block_index {
                // Persist the last block
                self.blocks.push(SimdBlock::from_usize_array(current_block));
                current_block = [0; SimdBlock::USIZE_COUNT];

                let gap = block_index - current_block_index;
                // Make it sparse as the overhead will be smaller than having empty blocks
                if gap > 1 {
                    // Create a new pseudo-offset
                    let root_block_offset = bit / SimdBlock::BITS;

                    pseudo = PseudoOffset {
                        blocks_offset: self.blocks.len() as u32,
                        root_bitset_offset: root_block_offset as u32,
                    };

                    // Maintain the invariant, marking the end of the sub-offsets
                    self.sub_offsets.push(pseudo);

                    current_block_index = root_block_offset;
                } else {
                    current_block_index += 1;
                }
            }

            current_block[sub_block % SimdBlock::USIZE_COUNT] |= 1 << remaining;
        }
        // Push the final block
        self.blocks.push(SimdBlock::from_usize_array(current_block));

        pseudo = PseudoOffset {
            blocks_offset: self.blocks.len() as u32,
            root_bitset_offset: 0,
        };
        // Maintain the invariant, marking the end of the sub-offsets
        self.sub_offsets.push(pseudo);
        
        // Maintain the invariant
        // We always have an empty sub_offset as the last item in the list to avoid branches
        self.offsets.push(self.sub_offsets.len() as u32 - 1);

        self.len() - 1
    }

    /// Retrieve the given `bit_set` from storage
    #[inline]
    pub fn get_set_ref(&self, bit_set: usize) -> SparseBitSetRef<'_> {
        assert!(
            bit_set < self.len(),
            "get_set at index {bit_set} is out of bounds due to length: {}",
            self.len()
        );
        // SAFETY: We assert that the length is sufficiently small.
        unsafe { self.get_s_set_ref_unchecked(bit_set) }
    }

    /// Return the length of the current collection
    #[inline]
    pub fn len(&self) -> usize {
        // We'll always have the last offset be there to mark the end of the `blocks` array.
        self.offsets.len() - 1
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_s_set_ref_unchecked(&self, bit_set: usize) -> SparseBitSetRef<'_> {
        let offset = *self.offsets.get_unchecked(bit_set);
        let next_offset = *self.offsets.get_unchecked(bit_set + 1);

        SparseBitSetRef {
            bitsets: SparseOffsets {
                offsets: self
                    .sub_offsets
                    .get_unchecked(offset as usize..=next_offset as usize),
            },
            blocks: &self.blocks,
        }
    }
}

#[derive(Debug)]
struct SparseOffsets<'a> {
    offsets: &'a [PseudoOffset],
}

#[derive(Debug)]
pub struct SparseBitSet<'a, T> {
    bitsets: SparseOffsets<'a>,
    blocks: T,
}

impl<'a> SparseBitSet<'a, &'a [SimdBlock]> {
    /// Return all indexes of set bits.
    pub fn ones(&self) -> impl DoubleEndedIterator<Item=usize> + 'a {
        self.bit_sets().flat_map(|bit_set| {
            let offset = bit_set.root_block_offset() * SimdBlock::BITS;
            match bit_set.sub_blocks().split_first() {
                Some((&first_block, rem)) => {
                    let (&last_block, rem) = rem.split_last().unwrap_or((&0, rem));
                    Ones {
                        bitset_front: first_block,
                        bitset_back: last_block,
                        block_idx_front: 0,
                        block_idx_back: (1 + rem.len()) * BITS,
                        remaining_blocks: rem.iter().copied(),
                        _phantom: Default::default(),
                    }
                }
                None => Ones {
                    bitset_front: 0,
                    bitset_back: 0,
                    block_idx_front: 0,
                    block_idx_back: 0,
                    remaining_blocks: [].iter().copied(),
                    _phantom: Default::default(),
                },
            }.map(move |i| i + offset)
        })
    }

    /// Return the last set in this bitset.
    ///
    /// This differs from [Self::borrow_bit_sets] that the lifetime of the returned iterator and sets is _not_ dependent on `self`,
    /// thus it can be obtained and stored alongside the bitset.
    #[inline]
    pub fn bit_sets(
        &self,
    ) -> impl ExactSizeIterator<Item = OffsetBitSetRef<'a>> + DoubleEndedIterator + 'a {
        let blocks = self.blocks;
        let offsets = self.bitsets.offsets;
        // SAFETY: We know `i` to always be in range of our sets
        (0..self.bit_sets_len()).map(move |i| unsafe { create_offset_bit_set(i, offsets, blocks) })
    }

    /// Return the last set in this bitset.
    ///
    /// This differs from [Self::borrow_last_set] that the lifetime of the returned set is _not_ dependent on `self`,
    /// thus it can be obtained and stored alongside the bitset.
    #[inline]
    fn get_last_set(&self) -> OffsetBitSetRef<'a> {
        // SAFETY: Guaranteed to be in range
        unsafe { self.get_set_ref_unchecked(self.bit_sets_len() - 1) }
    }

    #[inline]
    unsafe fn get_set_ref_unchecked(&self, bit_set: usize) -> OffsetBitSetRef<'a> {
        create_offset_bit_set(bit_set, self.bitsets.offsets, self.blocks)
    }
}

impl<'a, T: AsRef<[SimdBlock]>> SparseBitSet<'a, T> {
    /// Return the amount of [SimdBlock]s
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.simd_blocks().len()
    }

    pub(crate) fn bit_sets_len(&self) -> usize {
        // Last item is to mark the end of the block list
        self.bitsets.offsets.len() - 1
    }

    #[inline]
    pub fn root_block_offsets(&self) -> impl ExactSizeIterator<Item=usize> + DoubleEndedIterator + '_ {
        self.bitsets.offsets.iter().map(|off| off.root_bitset_offset as usize)
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        self.borrow_bit_sets().any(|bitset| {
            bitset.contains(bit)
        })
    }

    #[inline]
    pub(crate) fn borrow_bit_sets(
        &self,
    ) -> impl ExactSizeIterator<Item = OffsetBitSetRef<'_>> + DoubleEndedIterator {
        // SAFETY: We know `i` to always be in range of our sets
        (0..self.bit_sets_len()).map(move |i| unsafe { self.borrow_set_ref_unchecked(i) })
    }

    #[inline]
    fn borrow_last_set(&self) -> OffsetBitSetRef<'_> {
        // SAFETY: Guaranteed to be in range
        unsafe { self.borrow_set_ref_unchecked(self.bit_sets_len() - 1) }
    }

    #[inline]
    unsafe fn borrow_set_ref_unchecked(&self, bit_set: usize) -> OffsetBitSetRef<'_> {
        create_offset_bit_set(bit_set, self.bitsets.offsets, self.blocks.as_ref())
    }

    #[inline(always)]
    pub(crate) fn simd_blocks(&self) -> &[SimdBlock] {
        // SAFETY: The [SparseBitSet] maintains the invariant that it is never empty, and has a last `offsets` element
        // which holds the end of the `blocks` array. It is thus safe to directly index as below.

        unsafe {
            let first = self.bitsets.offsets.get_unchecked(0);
            let last = self
                .bitsets
                .offsets
                .get_unchecked(self.bitsets.offsets.len() - 1);
            self.blocks
                .as_ref()
                .get_unchecked(first.blocks_offset as usize..last.blocks_offset as usize)
        }
    }

    #[inline(always)]
    pub(crate) fn sub_blocks(&self) -> &[Block] {
        let simd_blocks = self.simd_blocks();
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangeable.
        unsafe {
            core::slice::from_raw_parts(
                simd_blocks.as_ptr().cast(),
                simd_blocks.len() * SimdBlock::USIZE_COUNT,
            )
        }
    }
}

/// Create an [OffsetBitSetRef] from a particular collection of bitsets.
///
/// # Safety
/// `bit_set` must be in range, and `bit_set + 1` must point to a valid `offset`.
#[inline(always)]
unsafe fn create_offset_bit_set<'a>(
    bit_set: usize,
    offsets: &'a [PseudoOffset],
    blocks: &'a [SimdBlock],
) -> OffsetBitSetRef<'a> {
    let offset = offsets.get_unchecked(bit_set);
    let next_offset = offsets.get_unchecked(bit_set + 1);
    crate::OffsetBitSet {
        root_block_offset: offset.root_bitset_offset,
        blocks: blocks
            .as_ref()
            .get_unchecked(offset.blocks_offset as usize..next_offset.blocks_offset as usize),
    }
}

struct ExactSizeFlatten<I, U: IntoIterator, F> {
    inner: FlatMap<I, U, F>,
    len: usize,
    consumed: usize,
}

impl<I, U: IntoIterator, F> ExactSizeFlatten<I, U, F> {
    pub fn new(expected_len: usize, flatten: FlatMap<I, U, F>) -> Self {
        Self {
            inner: flatten,
            len: expected_len,
            consumed: 0,
        }
    }
}

impl<I: Iterator, U: IntoIterator, F> Iterator for ExactSizeFlatten<I, U, F>
where
    F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        self.consumed += 1;
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }
}

impl<I: DoubleEndedIterator, U, F> DoubleEndedIterator for ExactSizeFlatten<I, U, F>
where
    F: FnMut(I::Item) -> U,
    U: IntoIterator<IntoIter: DoubleEndedIterator>,
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        self.consumed += 1;
        self.inner.next_back()
    }
    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, fold)
    }
}

impl<I, U, F> core::iter::FusedIterator for ExactSizeFlatten<I, U, F>
where
    I: core::iter::FusedIterator,
    U: IntoIterator,
    F: FnMut(I::Item) -> U,
{
}

impl<I: Iterator, U, F> ExactSizeIterator for ExactSizeFlatten<I, U, F>
where
    F: FnMut(I::Item) -> U,
    U: IntoIterator,
{
    fn len(&self) -> usize {
        self.len.saturating_sub(self.consumed)
    }
}

impl<'a, T: AsRef<[SimdBlock]>> BitSet for SparseBitSet<'a, T> {
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item = SimdBlock> + DoubleEndedIterator {
        let mut recent_root_offset = self.root_block_offset();
        let last_set = self.borrow_last_set();
        let final_len =
            last_set.root_block_offset as usize - recent_root_offset + last_set.blocks.len() + 1;
        
        ExactSizeFlatten::new(
            final_len,
            self.borrow_bit_sets().flat_map(move |b| {
                let to_pad = b.root_block_offset as usize - recent_root_offset;
                recent_root_offset = b.root_block_offset as usize;
                core::iter::repeat_n(SimdBlock::NONE, to_pad).chain(b.blocks.iter().copied())
            }),
        )
    }

    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item = Block> + DoubleEndedIterator {
        SimdToSubIter::new(self.as_simd_blocks())
    }

    fn bit_len(&self) -> usize {
        self.len() * SimdBlock::BITS
    }

    fn root_block_offset(&self) -> usize {
        self.bitsets.offsets[0].root_bitset_offset as usize
    }
}

// mod itr {
//     use core::marker::PhantomData;
//     use crate::generic::BitSet;
//     use crate::SimdBlock;
//     use crate::Block;
//     use crate::sparse::SparseBitSetRef;
//
//     #[derive(Debug)]
//     pub struct SparseOverlapIter<'a, I> {
//         sparse: &'a SparseBitSetRef<'a>,
//         current_bitset: u32,
//         current_root_block_offset: u32,
//         current_block_idx: u32,
//         overlap_state: Option<OverlapState>,
//         itr: I,
//         _phantom: PhantomData<&'a ()>,
//     }
//
//     pub fn new_overlap_simd<'a>(
//         left: &'a SparseBitSetRef<'a>,
//         right: &'a impl BitSet,
//     ) -> SparseOverlapIter<'a, impl DoubleEndedIterator<Item = (SimdBlock, SimdBlock)> + ExactSizeIterator + 'a>
//     {
//         let first_overlap = left.bit_sets().enumerate()
//             .map(|(idx, off)| (idx, calculate_overlap(&off, right)))
//             .filter(|(_, overlap)| overlap.is_valid()).next();
//
//         // No need to `.take()` on `other_bitset` as our previous slice takes care of that.
//         let itr = left
//             .as_simd_blocks()
//             .skip(overlap.left_offset)
//             .take(overlap.overlap_len)
//             .zip(right.as_simd_blocks().skip(overlap.right_offset));
//
//         SparseOverlapIter {
//             sparse: left,
//             current_bitset: first_overlap.as_ref().map(|f| f.0).unwrap_or_default() as u32,
//             current_root_block_offset: 0,
//             current_block_idx: 0,
//             overlap_state: first_overlap.map(|i| i.1),
//             itr,
//             _phantom: Default::default(),
//         }
//     }
//
//     pub fn new_overlap_sub_blocks<'a>(
//         left: &'a impl BitSet,
//         right: &'a impl BitSet,
//     ) -> SparseOverlapIter<'a, impl DoubleEndedIterator<Item = (Block, Block)> + ExactSizeIterator + 'a> {
//         let overlap = calculate_overlap(left, right);
//         // No need to `.take()` on `other_bitset` as our previous slice takes care of that.
//         let itr = left
//             .as_sub_blocks()
//             .skip(overlap.left_offset * SimdBlock::USIZE_COUNT)
//             .take(overlap.overlap_len * SimdBlock::USIZE_COUNT)
//             .zip(right.as_sub_blocks().skip(overlap.right_offset * SimdBlock::USIZE_COUNT));
//
//         SparseOverlapIter {
//             itr,
//             _phantom: Default::default(),
//         }
//     }
//
//     #[derive(Debug)]
//     pub struct OverlapState {
//         pub left_offset: usize,
//         pub right_offset: usize,
//         pub overlap_start: usize,
//         pub overlap_len: usize,
//     }
//
//     impl OverlapState {
//         pub fn is_valid(&self) -> bool {
//             self.overlap_start < self.overlap_len
//         }
//     }
//
//     #[inline]
//     pub fn calculate_overlap(left: &impl BitSet, right: &impl BitSet) -> OverlapState {
//         let self_start = left.root_block_offset();
//         let other_start = right.root_block_offset();
//
//         let overlap_start = self_start.max(other_start);
//         let overlap_end = left
//             .root_block_len()
//             .min(right.root_block_len());
//
//         let left_offset = overlap_start - self_start;
//         let right_offset = overlap_start - other_start;
//         let overlap_len = overlap_end.saturating_sub(overlap_start);
//
//         OverlapState {
//             left_offset,
//             right_offset,
//             overlap_start,
//             overlap_len,
//         }
//     }
//
//     #[inline]
//     pub fn overlap_start(left: &impl BitSet, right: &impl BitSet) -> usize {
//         let self_start = left.root_block_offset();
//         let other_start = right.root_block_offset();
//
//         self_start.max(other_start)
//     }
//
//     impl<'a, T, I: Iterator<Item = (T, T)>> Iterator for SparseOverlapIter<'a, I> {
//         type Item = (T, T);
//
//         #[inline]
//         fn next(&mut self) -> Option<Self::Item> {
//             self.itr.next()
//         }
//
//         #[inline]
//         fn size_hint(&self) -> (usize, Option<usize>) {
//             self.itr.size_hint()
//         }
//     }
//
//     impl<'a, T, I: ExactSizeIterator<Item = (T, T)>> ExactSizeIterator for SparseOverlapIter<'a, I> {
//         #[inline]
//         fn len(&self) -> usize {
//             self.itr.len()
//         }
//     }
//
//     impl<'a, T, I> DoubleEndedIterator for SparseOverlapIter<'a, I>
//     where
//         I: DoubleEndedIterator + Iterator<Item = (T, T)>,
//     {
//         #[inline]
//         fn next_back(&mut self) -> Option<Self::Item> {
//             self.itr.next_back()
//         }
//
//         #[inline]
//         fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
//             self.itr.nth_back(n)
//         }
//     }
//
//     impl<'a, T, I: FusedIterator<Item = (T, T)>> FusedIterator for SparseOverlapIter<'a, I> {}
// }


pub fn test_sparse_contains() {
    let mut coll = SparseBitSetCollection::new();

    let idx = coll.push_collection(&[1, 512]);
    let idx2 = coll.push_collection(&[128]);

    let set = coll.get_set_ref(idx);
    let set2 = coll.get_set_ref(idx2);

    assert!(set.contains(1));
    assert!(!set.contains(235));
    assert!(set.contains(512));
    assert!(set2.contains(128));
}

#[cfg(test)]
mod tests {
    use crate::generic::BitSet;
    use crate::sparse::SparseBitSetCollection;
    use crate::FixedBitSet;
    use alloc::vec::Vec;

    #[test]
    pub fn test_sparse() {
        let mut coll = SparseBitSetCollection::new();

        let idx = coll.push_collection(&[1, 512]);
        println!("Coll: {coll:?}\nIdx: {idx}");

        let set = coll.get_set_ref(idx);

        println!("Set: {set:?}\nItems: {:?}", set.root_block_offset());
        println!("Data: {:?}", set.as_simd_blocks().collect::<Vec<_>>());
    }

    #[test]
    pub fn test_sparse_subset() {
        let mut fset = FixedBitSet::with_capacity(1000);
        fset.insert_range(100..600);
        let mut coll = SparseBitSetCollection::new();

        let idx = coll.push_collection(&[1, 512]);
        let idx2 = coll.push_collection(&[128]);

        let set = coll.get_set_ref(idx);
        let set2 = coll.get_set_ref(idx2);

        assert!(!set.is_subset(&fset));
        assert!(set2.is_subset(&fset));
    }

    #[test]
    pub fn test_sparse_contains() {
        let mut coll = SparseBitSetCollection::new();

        let idx = coll.push_collection(&[1, 512]);
        let idx2 = coll.push_collection(&[128]);

        let set = coll.get_set_ref(idx);
        let set2 = coll.get_set_ref(idx2);

        assert!(set.contains(1));
        assert!(!set.contains(235));
        assert!(set.contains(512));
        assert!(set2.contains(128));
    }

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
        let left_cont = left.as_simd_blocks().collect::<Vec<_>>();
        // let mut itr = left.as_simd_blocks();
        // println!("T: {:?}", itr.next_back());
        // while let Some(i) = itr.next() {
        //     println!("CONT: {i:?}");
        // }
        // let right = base_collection.get_set_ref(right_idx);

        println!("ROOT: {:?}", base_collection);
        println!("CONTENT: {:?}", left_cont);
        let things = left.ones().collect::<Vec<_>>();
        let combined = left.lazy_and(&fset2);
        let combined2 = combined.ones().collect::<Vec<_>>();
        let comb_data = combined.as_simd_blocks().collect::<Vec<_>>();
        println!("STUFF: {combined2:?}\n{things:?}");
        println!("COMB: {:?}", comb_data);
        assert!(!combined.is_subset(&fset));
    }
}
