use crate::move_generation::mov::Move;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

const ENTRIES_PER_BUCKET: usize = 4;
const AGE_BITS: u64 = 0b11_1111;

#[derive(Debug)]
struct Entry {
    zobrist_key_xor_data: AtomicU64,
    data: AtomicU64,
}

/// Four entries sharing one cache line, so a probe touches a single line.
#[repr(align(64))]
struct Bucket {
    entries: [Entry; ENTRIES_PER_BUCKET],
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct EntryResult {
    pub score: i32,
    /// Static evaluation from the side to move's point of view (meaningless when in check)
    pub static_eval: i32,
    pub depth: u8,
    pub node_type: i8,
    pub best_move: Move,
}

impl EntryResult {
    pub fn new(depth: u8, score: i32, static_eval: i32, node_type: i8, best_move: Move) -> Self {
        Self {
            score,
            static_eval,
            depth,
            node_type,
            best_move,
        }
    }

    /// Layout, from the least significant bit: move (16), score (16), static eval (16),
    /// depth (8), node type (2), age (6).
    fn to_u64(self, age: u8) -> u64 {
        self.best_move.data as u64
            | ((self.score as i16 as u16 as u64) << 16)
            | ((self.static_eval as i16 as u16 as u64) << 32)
            | (self.depth as u64) << 48
            | ((self.node_type as u64) & 0b11) << 56
            | ((age as u64) & AGE_BITS) << 58
    }

    fn from_u64(data: u64) -> Self {
        let node_type = ((data >> 56) & 0b11) as i8;
        Self {
            best_move: Move { data: data as u16 },
            score: (data >> 16) as u16 as i16 as i32,
            static_eval: (data >> 32) as u16 as i16 as i32,
            depth: (data >> 48) as u8,
            // Two-bit two's complement: 0b11 is -1
            node_type: if node_type == 0b11 { -1 } else { node_type },
        }
    }
}

fn age_of(data: u64) -> u8 {
    ((data >> 58) & AGE_BITS) as u8
}

pub struct TranspositionTable {
    size: usize,
    table: Vec<Bucket>,
    /// Bumped every search so entries from earlier searches are replaced first
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024 / size_of::<Bucket>()).max(1);
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Bucket {
                entries: std::array::from_fn(|_| Entry {
                    zobrist_key_xor_data: AtomicU64::new(0),
                    data: AtomicU64::new(0),
                }),
            });
        }
        Self {
            size,
            table,
            age: AtomicU8::new(0),
        }
    }

    pub fn new_search(&self) {
        let age = self.age.load(Ordering::Relaxed);
        self.age
            .store(((age as u64 + 1) & AGE_BITS) as u8, Ordering::Relaxed);
    }

    pub fn get(&self, zobrist_key: u64) -> Option<EntryResult> {
        let bucket = &self.table[self.index(zobrist_key)];
        for entry in &bucket.entries {
            let data = entry.data.load(Ordering::Relaxed);
            let key_xor_data = entry.zobrist_key_xor_data.load(Ordering::Relaxed);
            if key_xor_data ^ data == zobrist_key && data != 0 {
                return Some(EntryResult::from_u64(data));
            }
        }
        None
    }

    pub fn insert(&self, zobrist_key: u64, mut entry_result: EntryResult) {
        let bucket = &self.table[self.index(zobrist_key)];
        let age = self.age.load(Ordering::Relaxed);

        let mut replace = 0;
        let mut worst_value = i32::MAX;
        for (i, entry) in bucket.entries.iter().enumerate() {
            let data = entry.data.load(Ordering::Relaxed);
            let key_xor_data = entry.zobrist_key_xor_data.load(Ordering::Relaxed);
            if data == 0 && key_xor_data == 0 {
                replace = i;
                break;
            }
            if key_xor_data ^ data == zobrist_key {
                let existing = EntryResult::from_u64(data);
                // Keep a much deeper result for the same position from this search, unless
                // the new one is exact
                if entry_result.node_type != 0
                    && age_of(data) == age
                    && entry_result.depth as i32 + 4 < existing.depth as i32
                {
                    return;
                }
                if entry_result.best_move == Move::null() {
                    entry_result.best_move = existing.best_move;
                }
                replace = i;
                break;
            }
            // Otherwise evict the shallowest entry, treating old entries as shallower
            let age_difference = ((age as u64).wrapping_sub(age_of(data) as u64) & AGE_BITS) as i32;
            let value = EntryResult::from_u64(data).depth as i32 - 8 * age_difference;
            if value < worst_value {
                worst_value = value;
                replace = i;
            }
        }

        let entry = &bucket.entries[replace];
        let data = entry_result.to_u64(age);
        entry.data.store(data, Ordering::Relaxed);
        entry
            .zobrist_key_xor_data
            .store(zobrist_key ^ data, Ordering::Relaxed);
    }

    /// Maps a key onto the table with a multiply-shift instead of a 64-bit division.
    #[inline(always)]
    fn index(&self, zobrist_key: u64) -> usize {
        ((zobrist_key as u128 * self.size as u128) >> 64) as usize
    }

    pub fn clear(&self) {
        for bucket in &self.table {
            for entry in &bucket.entries {
                entry.zobrist_key_xor_data.store(0, Ordering::Relaxed);
                entry.data.store(0, Ordering::Relaxed);
            }
        }
        self.age.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use crate::move_generation::mov::Move;
    use crate::search_threads::transposition_table::{EntryResult, TranspositionTable};

    #[test]
    fn test_load_unload() {
        let entry_result = EntryResult::new(4, 10, -25, 0, Move::left_en_passant(10));
        let table = TranspositionTable::new(1);
        table.insert(1, entry_result);
        assert_eq!(table.get(1), Some(entry_result));
    }

    #[test]
    fn negative_scores_and_bounds_round_trip() {
        for node_type in [-1, 0, 1] {
            let entry_result = EntryResult::new(7, -31_950, 1234, node_type, Move::regular(1, 2));
            let table = TranspositionTable::new(1);
            table.insert(42, entry_result);
            assert_eq!(table.get(42), Some(entry_result));
        }
    }

    #[test]
    fn much_deeper_entry_is_not_replaced_by_shallow_bound() {
        let table = TranspositionTable::new(1);
        let deep = EntryResult::new(12, 100, 0, 0, Move::left_en_passant(10));
        let shallow = EntryResult::new(4, -100, 0, 1, Move::null());

        table.insert(5, deep);
        table.insert(5, shallow);

        assert_eq!(table.get(5), Some(deep));
    }

    #[test]
    fn colliding_positions_share_a_bucket() {
        let table = TranspositionTable::new(1);
        // Keys that map to the same bucket: the top bits pick the bucket
        let keys = [1u64, 2, 3, 4];
        for (i, &key) in keys.iter().enumerate() {
            table.insert(
                key,
                EntryResult::new(i as u8 + 1, i as i32, 0, 0, Move::null()),
            );
        }
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(table.get(key).unwrap().score, i as i32);
        }
    }
}
