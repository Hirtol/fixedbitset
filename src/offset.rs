use crate::{Block, FixedBitSet};
use crate::SimdBlock;
use alloc::vec::Vec;

pub type OffsetBitSetOwned = OffsetBitSet<Vec<SimdBlock>>;
pub type OffsetBitSetRef<'a> = OffsetBitSet<&'a [SimdBlock]>;
pub type OffsetBitSetMut<'a> = OffsetBitSet<&'a mut [SimdBlock]>;

pub struct OffsetBitSetCollection {
    /// Tracks the offset in [SimdBlock] counts into the `block` array.
    /// 
    /// # Invariant
    /// There is always a last PseudoOffset to mark the end of the array
    pub offsets: Vec<PseudoOffset>,
    pub blocks: Vec<SimdBlock>
}

#[derive(Default)]
struct PseudoOffset {
    blocks_offset: u32,
    root_bitset_offset: u32,
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
        assert!(!bits.is_empty(), "Need a non-empty set");
        let root_block_offset = bits[0] / SimdBlock::BITS;

        let blocks_offset = self.blocks.len();
        let mut current_block_index = root_block_offset;
        let mut current_block = [0; SimdBlock::USIZE_COUNT];
        for &bit in bits {
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

        // Note the offset
        let new_entry = self.offsets.last_mut().expect("Impossible invariant violation");
        new_entry.blocks_offset = blocks_offset as u32;
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
        let extensions: &[SimdBlock] = unsafe {
            core::slice::from_raw_parts(blocks.as_ptr().cast(), simd_blocks)
        };
        self.blocks.extend_from_slice(extensions);
        // Deal with the remainder
        if rem > 0 {
            let mut modify = [0; SimdBlock::USIZE_COUNT];
            modify.copy_from_slice(&blocks[safe_blocks..]);
            self.blocks.push(SimdBlock::from_usize_array(modify));
        }
        
        // Note the offset
        let new_entry = self.offsets.last_mut().expect("Impossible invariant violation");
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
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_ref_unchecked(bit_set)
        }
    }

    /// Retrieve the given `bit_set` from storage
    #[inline]
    pub fn get_set_ref_mut(&mut self, bit_set: usize) -> OffsetBitSetMut<'_> {
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_ref_mut_unchecked(bit_set)
        }
    }

    #[inline]
    pub fn get_set_as_slice(&self, bit_set: usize) -> &[SimdBlock] {
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_as_slice_unchecked(bit_set)
        }
    }

    #[inline]
    pub fn get_set_as_slice_mut(&mut self, bit_set: usize) -> &mut [SimdBlock] {
        assert!(bit_set < self.len(), "get_set at index {bit_set} is out of bounds due to length: {}", self.len());
        // SAFETY: We assert that the length is sufficiently small.
        unsafe {
            self.get_set_as_slice_mut_unchecked(bit_set)
        }
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
            blocks: self.blocks.get_unchecked(offset.blocks_offset as usize..next_offset.blocks_offset as usize),
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
            blocks: self.blocks.get_unchecked_mut(offset.blocks_offset as usize..next_offset.blocks_offset as usize),
        }
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_as_slice_mut_unchecked(&mut self, bit_set: usize) -> &mut [SimdBlock] {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        self.blocks.get_unchecked_mut(offset.blocks_offset as usize..next_offset.blocks_offset as usize)
    }

    /// # Safety
    /// `bit_set` must not exceed the [Self::len()]
    #[inline]
    pub unsafe fn get_set_as_slice_unchecked(&self, bit_set: usize) -> &[SimdBlock] {
        let offset = self.offsets.get_unchecked(bit_set);
        let next_offset = self.offsets.get_unchecked(bit_set + 1);
        self.blocks.get_unchecked(offset.blocks_offset as usize..next_offset.blocks_offset as usize)
    }
}

#[derive(Debug)]
pub struct OffsetBitSet<T> {
    root_block_offset: u32,
    blocks: T
}

impl<T: AsRef<[SimdBlock]>> OffsetBitSet<T> {
    pub fn as_fixed_bit_set(&self, bits: usize) -> FixedBitSet {
        let bits_to_start = self.root_block_offset as usize * SimdBlock::BITS;
        let total_bits = bits_to_start + self.simd_blocks().len() * SimdBlock::BITS;
        assert!(total_bits < bits, "Creating a FixedBitSet out of an OffsetBitSet requires the total `bits` ({bits}) count to be larger than the OffsetBitSet's size ({total_bits})");
        
        let sblock_count = self.root_block_offset as usize * SimdBlock::USIZE_COUNT;
        let repeat = core::iter::repeat_n(0, sblock_count)
            .chain(self.simd_blocks().iter().flat_map(|v| v.into_usize_array()))
            .chain(core::iter::repeat(0));
        FixedBitSet::with_capacity_and_blocks(bits, repeat)
    }
    
    #[inline(always)]
    fn simd_blocks(&self) -> &[SimdBlock] {
        self.blocks.as_ref()
    }
    
    #[inline(always)]
    fn blocks(&self) -> &[Block] {
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangable.
        unsafe {
            core::slice::from_raw_parts(self.simd_blocks().as_ptr().cast(), self.simd_blocks().len() * SimdBlock::USIZE_COUNT)
        }
    }
}

impl<T: AsMut<[SimdBlock]> + AsRef<[SimdBlock]>> OffsetBitSet<T> {
    #[inline(always)]
    fn simd_blocks_mut(&mut self) -> &mut [SimdBlock] {
        self.blocks.as_mut()
    }

    #[inline(always)]
    fn blocks_mut(&mut self) -> &mut [Block] {
        // SAFETY: The representations of SimdBlock and Block are guaranteed to be interchangable.
        unsafe {
            core::slice::from_raw_parts_mut(self.simd_blocks_mut().as_mut_ptr().cast(), self.simd_blocks_mut().len() * SimdBlock::USIZE_COUNT)
        }
    }
}

impl<'a> OffsetBitSet<&'a [SimdBlock]> {
    #[inline(always)]
    pub fn from_fixed_set(offset_start: u32, offset_end: u32, other: &'a FixedBitSet) -> Self {
        Self {
            root_block_offset: offset_start,
            blocks: &other.as_simd_slice()[offset_start as usize..offset_end as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::offset::OffsetBitSetCollection;
    use crate::SimdBlock;

    #[test]
    pub fn test_push_offset_set() {
        let mut base_collection = OffsetBitSetCollection::new();
        
        let index = base_collection.push_collection(&[100, 127, 256]);
        
        let set = base_collection.get_set_ref(index);
        let testing_set = set.as_fixed_bit_set(400);
        assert_eq!(set.blocks.len(), 3);
        println!("Set: {set:#?}");
        println!("Blocks: {:?}", set.blocks());
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
        println!("Set: {set:#?} - {}", SimdBlock::BITS);
        println!("Blocks: {:?}", set.blocks());
        for i in testing_set.ones() {
            println!("Ones: {i}");
        }
        assert!(testing_set.contains(128));
        assert!(testing_set.contains(192));
        
    }
}