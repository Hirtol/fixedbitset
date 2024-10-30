use crate::generic::BitSet;
use crate::SimdBlock;
use crate::{Block, FixedBitSet, IndexRange};
use alloc::vec::Vec;

pub mod iter;
pub mod sparse;
pub mod iter_exact;

pub type OffsetBitSetOwned = OffsetBitSet<Vec<SimdBlock>>;
pub type OffsetBitSetRef<'a> = OffsetBitSet<&'a [SimdBlock]>;
pub type OffsetBitSetMut<'a> = OffsetBitSet<&'a mut [SimdBlock]>;

#[derive(Debug, Clone)]
pub struct OffsetBitSetCollection {
    /// Tracks the offset in [SimdBlock] counts into the `block` array.
    ///
    /// # Invariant
    /// There is always a last PseudoOffset to mark the end of the array
    offsets: Vec<PseudoOffset>,
    blocks: Vec<SimdBlock>,
}

#[derive(Default, Debug, Clone, Copy)]
struct PseudoOffset {
    blocks_offset: u32,
    root_bitset_offset: u32,
}

impl Default for OffsetBitSetCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl OffsetBitSetCollection {
    pub fn new() -> Self {
        Self {
            offsets: alloc::vec![PseudoOffset::default()],
            blocks: Vec::new(),
        }
    }

    /// Push a _sorted_ collection of bits which are expected to be enabled as a new [OffsetBitSet].
    ///
    /// A base offset will automatically be inferred based on the first item in the list
    ///
    /// # Returns
    /// The index to access the new [OffsetBitSet]
    pub fn push_collection(&mut self, bits: &[usize]) -> usize {
        self.push_collection_itr(bits.iter().copied())
    }

    /// Push a _sorted_ collection of bits which are expected to be enabled as a new [OffsetBitSet].
    ///
    /// A base offset will automatically be inferred based on the first item in the list
    ///
    /// # Returns
    /// The index to access the new [OffsetBitSet]
    pub fn push_collection_itr(&mut self, mut bits: impl ExactSizeIterator<Item = usize>) -> usize {
        let Some(first_item) = bits.next() else {
            panic!("Need a non-empty set");
        };
        let root_block_offset = first_item / SimdBlock::BITS;

        let mut current_block_index = root_block_offset;
        let mut current_block = [0; SimdBlock::USIZE_COUNT];
        for bit in core::iter::once(first_item).chain(bits) {
            let (sub_block, remaining) = crate::div_rem(bit, crate::BITS);
            let block_index = sub_block / SimdBlock::USIZE_COUNT;

            // Fill the space between this block and the next, can result in a lot of empty spots for discontiguous sets, but that's the trade-off
            while block_index > current_block_index {
                self.blocks.push(SimdBlock::from_usize_array(current_block));
                current_block = [0; SimdBlock::USIZE_COUNT];
                current_block_index += 1;
            }

            current_block[sub_block % SimdBlock::USIZE_COUNT] |= 1 << remaining;
        }
        // Push the final block
        self.blocks.push(SimdBlock::from_usize_array(current_block));

        // Note the offset, `blocks_offset` was already correct from the previous iteration.
        let new_entry = self
            .offsets
            .last_mut()
            .expect("Impossible invariant violation");
        new_entry.root_bitset_offset = root_block_offset as u32;
        // Maintain the invariant
        self.offsets.push(PseudoOffset {
            blocks_offset: self.blocks.len() as u32,
            root_bitset_offset: 0,
        });

        self.len() - 1
    }

    /// # Safety
    ///
    /// Requires `blocks` to be aligned to [SimdBlock] alignment, best achieved by just allocating a fresh `Vec`.
    ///
    /// # Arguments
    /// * `root_block_offset` - The offset in the base [FixedBitSet] in [SimdBlock]s
    pub unsafe fn push_set(&mut self, root_block_offset: usize, blocks: &[Block]) -> usize {
        let (simd_blocks, rem) = crate::div_rem(blocks.len(), SimdBlock::USIZE_COUNT);
        let safe_blocks = simd_blocks * SimdBlock::USIZE_COUNT;
        let blocks_offset = self.blocks.len();

        // SAFETY: We can assume that the layout between the SIMD types and an array representation are the same
        let extensions: &[SimdBlock] =
            unsafe { core::slice::from_raw_parts(blocks.as_ptr().cast(), simd_blocks) };
        self.blocks.extend_from_slice(extensions);
        // Deal with the remainder
        if rem > 0 {
            let mut modify = [0; SimdBlock::USIZE_COUNT];
            modify.copy_from_slice(&blocks[safe_blocks..]);
            self.blocks.push(SimdBlock::from_usize_array(modify));
        }

        // Note the offset
        let new_entry = self
            .offsets
            .last_mut()
            .expect("Impossible invariant violation");
        new_entry.blocks_offset = blocks_offset as u32;
        new_entry.root_bitset_offset = root_block_offset as u32;
        // Maintain the invariant
        self.offsets.push(PseudoOffset {
            blocks_offset: self.blocks.len() as u32,
            root_bitset_offset: 0,
        });

        self.len() - 1
    }

    /// Retrieve the given `bit_set` from storage
    #[inline]
    pub fn get_set_ref(&self, bit_set: usize) -> OffsetBitSetRef<'_> {
        assert!(
            bit_set < self.len(),
            "get_set at index {bit_set} is out of bounds due to length: {}",
            self.len()
        );
        // SAFETY: We assert that the length is sufficiently small.
        unsafe { self.get_set_ref_unchecked(bit_set) }
    }

    /// Retrieve the given `bit_set` from storage
    #[inline]
    pub fn get_set_ref_mut(&mut self, bit_set: usize) -> OffsetBitSetMut<'_> {
        assert!(
            bit_set < self.len(),
            "get_set at index {bit_set} is out of bounds due to length: {}",
            self.len()
        );
        // SAFETY: We assert that the length is sufficiently small.
        unsafe { self.get_set_ref_mut_unchecked(bit_set) }
    }

    #[inline]
    pub fn get_set_as_slice(&self, bit_set: usize) -> &[SimdBlock] {
        assert!(
            bit_set < self.len(),
            "get_set at index {bit_set} is out of bounds due to length: {}",
            self.len()
        );
        // SAFETY: We assert that the length is sufficiently small.
        unsafe { self.get_set_as_slice_unchecked(bit_set) }
    }

    #[inline]
    pub fn get_set_as_slice_mut(&mut self, bit_set: usize) -> &mut [SimdBlock] {
        assert!(
            bit_set < self.len(),
            "get_set at index {bit_set} is out of bounds due to length: {}",
            self.len()
        );
        // SAFETY: We assert that the length is sufficiently small.
        unsafe { self.get_set_as_slice_mut_unchecked(bit_set) }
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
    pub unsafe fn get_set_ref_unchecked(&self, bit_set: usize) -> OffsetBitSetRef<'_> {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        OffsetBitSet {
            root_block_offset: offset.root_bitset_offset,
            blocks: self
                .blocks
                .get_unchecked(offset.blocks_offset as usize..next_offset.blocks_offset as usize),
        }
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_ref_mut_unchecked(&mut self, bit_set: usize) -> OffsetBitSetMut<'_> {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        OffsetBitSet {
            root_block_offset: offset.root_bitset_offset,
            blocks: self.blocks.get_unchecked_mut(
                offset.blocks_offset as usize..next_offset.blocks_offset as usize,
            ),
        }
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_as_slice_mut_unchecked(&mut self, bit_set: usize) -> &mut [SimdBlock] {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        self.blocks
            .get_unchecked_mut(offset.blocks_offset as usize..next_offset.blocks_offset as usize)
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_as_slice_unchecked(&self, bit_set: usize) -> &[SimdBlock] {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        self.blocks
            .get_unchecked(offset.blocks_offset as usize..next_offset.blocks_offset as usize)
    }
}

#[derive(Debug)]
pub struct OffsetBitSet<T> {
    root_block_offset: u32,
    pub blocks: T,
}

impl<T: AsRef<[SimdBlock]>> OffsetBitSet<T> {

    /// Return a reference to this [OffsetBitSet]
    pub fn as_ref(&self) -> OffsetBitSetRef<'_> {
        OffsetBitSetRef {
            root_block_offset: self.root_block_offset,
            blocks: self.blocks.as_ref(),
        }
    }

    /// Return an independent, owned, [OffsetBitSet]
    pub fn to_owned(&self) -> OffsetBitSetOwned {
        OffsetBitSetOwned {
            root_block_offset: self.root_block_offset,
            blocks: self.ref_simd_blocks().to_vec(),
        }
    }

    /// Return the amount of [SimdBlock]s
    #[inline(always)]
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.as_simd_blocks().len()
    }

    /// Return the length of the OffsetBitSet if it was instead a full [FixedBitSet]
    #[inline(always)]
    #[allow(dead_code)]
    fn root_block_len(&self) -> usize {
        self.root_block_offset as usize + self.len()
    }

    #[inline(always)]
    pub(crate) fn ref_simd_blocks(&self) -> &[SimdBlock] {
        self.blocks.as_ref()
    }

    #[inline(always)]
    pub(crate) fn ref_sub_blocks(&self) -> &[Block] {
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangeable.
        unsafe {
            core::slice::from_raw_parts(
                self.ref_simd_blocks().as_ptr().cast(),
                self.ref_simd_blocks().len() * SimdBlock::USIZE_COUNT,
            )
        }
    }
}

impl<T: AsMut<[SimdBlock]> + AsRef<[SimdBlock]>> OffsetBitSet<T> {
    #[allow(dead_code)]
    #[inline(always)]
    fn simd_blocks_mut(&mut self) -> &mut [SimdBlock] {
        self.blocks.as_mut()
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn sub_blocks_mut(&mut self) -> &mut [Block] {
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangeable.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.simd_blocks_mut().as_mut_ptr().cast(),
                self.simd_blocks_mut().len() * SimdBlock::USIZE_COUNT,
            )
        }
    }
}

impl<'a> OffsetBitSet<&'a [SimdBlock]> {
    #[inline(always)]
    pub(crate) fn simd_blocks(&self) -> &'a [SimdBlock] {
        self.blocks
    }

    #[inline(always)]
    pub(crate) fn sub_blocks(&self) -> &'a [Block] {
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangeable.
        unsafe {
            core::slice::from_raw_parts(
                self.blocks.as_ptr().cast(),
                self.blocks.len() * SimdBlock::USIZE_COUNT,
            )
        }
    }
}

impl<'a> OffsetBitSetRef<'a> {
    #[inline(always)]
    pub fn from_fixed_set<I: IndexRange>(block_offsets: I, other: &'a FixedBitSet) -> Self {
        let start = block_offsets.start().unwrap_or(0);
        let end = block_offsets
            .end()
            .unwrap_or_else(|| other.simd_block_len());
        Self {
            root_block_offset: start as u32,
            blocks: &other.as_simd_slice()[start..end],
        }
    }
}

impl<T: AsRef<[SimdBlock]>> BitSet for OffsetBitSet<T> {
    #[inline(always)]
    fn as_simd_blocks(&self) -> impl ExactSizeIterator<Item = SimdBlock> + DoubleEndedIterator {
        self.ref_simd_blocks().iter().copied()
    }

    #[inline(always)]
    fn as_sub_blocks(&self) -> impl ExactSizeIterator<Item = Block> + DoubleEndedIterator {
        self.ref_sub_blocks().iter().copied()
    }

    #[inline(always)]
    fn bit_len(&self) -> usize {
        self.as_simd_blocks().len() * SimdBlock::BITS
    }

    #[inline(always)]
    fn root_block_offset(&self) -> usize {
        self.root_block_offset as usize
    }
}

pub fn test_subset() {
    let mut base_collection = OffsetBitSetCollection::new();

    // Safety: The vec is guaranteed to be aligned.
    let index = base_collection.push_collection(&[128, 129, 256]);
    let other = base_collection.push_collection(&[129, 256]);

    let set = base_collection.get_set_ref(index);
    let other = base_collection.get_set_ref(other);
    assert!(other.is_subset(&set));

    let mut fixed = set.as_fixed_bit_set(400);
    assert!(other.is_subset(&fixed));
    fixed.remove(129);
    assert!(!other.is_subset(&fixed));

    let index = base_collection.push_collection(&[3, 128, 129, 256]);
    let other_new = base_collection.push_collection(&[3, 256]);

    let other_new = base_collection.get_set_ref(other_new);
    let set = base_collection.get_set_ref(index);
    assert!(other_new.is_subset(&set));
}

#[cfg(test)]
mod tests {
    use crate::generic::BitSet;
    use crate::offset::OffsetBitSetCollection;
    use alloc::vec::Vec;

    #[test]
    pub fn test_push_offset_set() {
        let mut base_collection = OffsetBitSetCollection::new();

        let index = base_collection.push_collection(&[100, 127, 256]);
        let other_idx = base_collection.push_collection(&[256]);

        let set = base_collection.get_set_ref(index);
        assert_eq!(set.blocks.len(), 3);
        assert_eq!(set.ones().collect::<Vec<_>>(), vec![100, 127, 256]);

        let other = base_collection.get_set_ref(other_idx);
        assert_eq!(other.ones().collect::<Vec<_>>(), vec![256]);

        let testing_set = set.as_fixed_bit_set(400);
        assert!(testing_set.contains(100));
        assert!(testing_set.contains(127));
        assert!(testing_set.contains(256));
        assert!(!testing_set.contains(255));
    }

    #[test]
    pub fn test_push_offset() {
        let mut base_collection = OffsetBitSetCollection::new();

        // Safety: The vec is guaranteed to be aligned.
        let index = unsafe { base_collection.push_set(1, &vec![1, 1]) };

        let set = base_collection.get_set_ref(index);
        let testing_set = set.as_fixed_bit_set(400);
        assert_eq!(set.blocks.len(), 1);
        assert!(testing_set.contains(128));
        assert!(testing_set.contains(192));
    }

    #[test]
    pub fn test_overlap() {
        let mut base_collection = OffsetBitSetCollection::new();

        // Safety: The vec is guaranteed to be aligned.
        let index = base_collection.push_collection(&[3, 128, 129, 256]);
        let other = base_collection.push_collection(&[129, 256]);

        let set = base_collection.get_set_ref(index);
        let other = base_collection.get_set_ref(other);

        assert!(set.has_overlap(&other));
        let overlap = set.overlap(&other).collect::<Vec<_>>();
        assert_eq!(overlap.len(), 2);
        println!("Overlap: {overlap:?}");

        let other_new = base_collection.push_collection(&[256]);
        let other_new = base_collection.get_set_ref(other_new);

        let set = base_collection.get_set_ref(index);
        let overlap = set.overlap(&other_new).collect::<Vec<_>>();
        println!("Overlap: {overlap:?}");

        assert_eq!(overlap.len(), 1);
    }

    #[test]
    pub fn test_subset() {
        let mut base_collection = OffsetBitSetCollection::new();

        // Safety: The vec is guaranteed to be aligned.
        let index = base_collection.push_collection(&[128, 129, 256]);
        let other = base_collection.push_collection(&[129, 256]);

        let set = base_collection.get_set_ref(index);
        let other = base_collection.get_set_ref(other);
        assert!(other.is_subset(&set));

        let mut fixed = set.as_fixed_bit_set(400);
        assert!(other.is_subset(&fixed));
        fixed.remove(129);
        assert!(!other.is_subset(&fixed));

        let index = base_collection.push_collection(&[3, 128, 129, 256]);
        let other_new = base_collection.push_collection(&[3, 256]);

        let other_new = base_collection.get_set_ref(other_new);
        let set = base_collection.get_set_ref(index);
        assert!(other_new.is_subset(&set));
    }
}
