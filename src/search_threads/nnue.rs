//! NNUE evaluation: a (768 -> HIDDEN) x 2 perspective network with SCReLU and output buckets
//! chosen by piece count. The accumulators are updated incrementally by the search.

use crate::move_generation::Board;

pub const INPUTS: usize = 768;
pub const HIDDEN: usize = 256;
pub const OUTPUT_BUCKETS: usize = 8;
/// Quantization of the accumulator (and the [0, 1] SCReLU range)
pub const QA: i32 = 255;
/// Quantization of the output weights
pub const QB: i32 = 64;
/// Network output 1.0 is this many centipawns
pub const SCALE: i32 = 400;

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [[i16; HIDDEN]; INPUTS],
    pub feature_bias: [i16; HIDDEN],
    /// Per bucket: side-to-move half, then the other side's half
    pub output_weights: [[i16; 2 * HIDDEN]; OUTPUT_BUCKETS],
    pub output_bias: [i16; OUTPUT_BUCKETS],
}

/// The trained network, stored as the raw bytes of `Network` (little-endian, padding
/// included) as written by `HydraChess train`.
static NET: Network = unsafe { std::mem::transmute(*include_bytes!("hydra.nnue")) };

#[inline(always)]
pub fn network() -> &'static Network {
    &NET
}

/// Feature index of a piece seen from `perspective` (0 white, 1 black).
#[inline(always)]
pub fn feature(perspective: usize, color: usize, piece: usize, square: usize) -> usize {
    if perspective == 0 {
        (color != 0) as usize * 384 + piece * 64 + square
    } else {
        (color != 1) as usize * 384 + piece * 64 + (square ^ 56)
    }
}

#[inline(always)]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2)) as usize / 4).min(OUTPUT_BUCKETS - 1)
}

/// Both perspectives' hidden layer sums. White is 0.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub values: [[i16; HIDDEN]; 2],
}

impl Accumulator {
    pub fn from_board(board: &Board) -> Self {
        let net = network();
        let mut acc = Self {
            values: [net.feature_bias; 2],
        };
        for_each_piece(board, |color, piece, square| {
            acc.add(net, color, piece, square);
        });
        acc
    }

    #[inline(always)]
    fn add(&mut self, net: &Network, color: usize, piece: usize, square: usize) {
        for perspective in 0..2 {
            let weights = &net.feature_weights[feature(perspective, color, piece, square)];
            for (v, w) in self.values[perspective].iter_mut().zip(weights) {
                *v += *w;
            }
        }
    }

    /// Applies the difference between two boards (one move apart), given as removed and added
    /// (color, piece, square) lists. Reads the parent once and writes the child once.
    #[inline(always)]
    pub fn update(&mut self, parent: &Self, removed: &[(usize, usize, usize)], added: &[(usize, usize, usize)]) {
        let net = network();
        for perspective in 0..2 {
            let row = |&(color, piece, square): &(usize, usize, usize)| {
                &net.feature_weights[feature(perspective, color, piece, square)]
            };
            let out = &mut self.values[perspective];
            let from = &parent.values[perspective];
            match (removed.len(), added.len()) {
                (1, 1) => {
                    let (s0, a0) = (row(&removed[0]), row(&added[0]));
                    for j in 0..HIDDEN {
                        out[j] = from[j] + a0[j] - s0[j];
                    }
                }
                (2, 1) => {
                    let (s0, s1, a0) = (row(&removed[0]), row(&removed[1]), row(&added[0]));
                    for j in 0..HIDDEN {
                        out[j] = from[j] + a0[j] - s0[j] - s1[j];
                    }
                }
                (2, 2) => {
                    let (s0, s1) = (row(&removed[0]), row(&removed[1]));
                    let (a0, a1) = (row(&added[0]), row(&added[1]));
                    for j in 0..HIDDEN {
                        out[j] = from[j] + a0[j] + a1[j] - s0[j] - s1[j];
                    }
                }
                _ => {
                    out.copy_from_slice(from);
                    for feature in added {
                        let weights = row(feature);
                        for j in 0..HIDDEN {
                            out[j] += weights[j];
                        }
                    }
                    for feature in removed {
                        let weights = row(feature);
                        for j in 0..HIDDEN {
                            out[j] -= weights[j];
                        }
                    }
                }
            }
        }
    }

    /// Evaluation in centipawns from the side to move's point of view.
    #[inline(always)]
    pub fn evaluate(&self, white_to_move: bool, piece_count: u32) -> i32 {
        let net = network();
        let bucket = output_bucket(piece_count);
        let (us, them) = if white_to_move { (0, 1) } else { (1, 0) };
        let weights = &net.output_weights[bucket];
        let sum = screlu_dot(&self.values[us], &weights[..HIDDEN])
            + screlu_dot(&self.values[them], &weights[HIDDEN..]);
        (sum / QA + net.output_bias[bucket] as i32) * SCALE / (QA * QB)
    }
}

#[inline(always)]
fn screlu_dot(values: &[i16], weights: &[i16]) -> i32 {
    // clamp(v)^2 * w = (clamp(v) * w) * clamp(v); the first product fits in an i16 because
    // |w| <= 127 and v <= 255
    let mut sum = [0i32; 16];
    for (v_chunk, w_chunk) in values.chunks_exact(16).zip(weights.chunks_exact(16)) {
        for i in 0..16 {
            let v = v_chunk[i].clamp(0, QA as i16);
            let vw = v.wrapping_mul(w_chunk[i]);
            sum[i] += vw as i32 * v as i32;
        }
    }
    sum.iter().sum()
}

/// Calls `f(color, piece type, square)` for every piece on the board.
#[inline(always)]
pub fn for_each_piece(board: &Board, mut f: impl FnMut(usize, usize, usize)) {
    let types = [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    ];
    for (color, side) in [board.white, board.black].into_iter().enumerate() {
        for (piece, &bitboard) in types.iter().enumerate() {
            let mut bits = bitboard & side;
            while bits != 0 {
                let square = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                f(color, piece, square);
            }
        }
    }
}

/// The 12 piece bitboards, white pawn .. black king, to diff before and after a move.
#[derive(Clone, Copy)]
pub struct PieceBoards([u64; 12]);

impl PieceBoards {
    #[inline(always)]
    pub fn of(board: &Board) -> Self {
        let types = [
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
        ];
        let mut boards = [0; 12];
        for piece in 0..6 {
            boards[piece] = types[piece] & board.white;
            boards[piece + 6] = types[piece] & board.black;
        }
        Self(boards)
    }

    /// Fills the removed and added lists going from `self` to `after`; returns their lengths.
    #[inline(always)]
    pub fn diff(
        &self,
        after: &Self,
        removed: &mut [(usize, usize, usize); 4],
        added: &mut [(usize, usize, usize); 4],
    ) -> (usize, usize) {
        let (mut r, mut a) = (0, 0);
        for i in 0..12 {
            let changed = self.0[i] ^ after.0[i];
            if changed == 0 {
                continue;
            }
            let (color, piece) = (i / 6, i % 6);
            let mut gone = self.0[i] & changed;
            while gone != 0 {
                removed[r] = (color, piece, gone.trailing_zeros() as usize);
                r += 1;
                gone &= gone - 1;
            }
            let mut new = after.0[i] & changed;
            while new != 0 {
                added[a] = (color, piece, new.trailing_zeros() as usize);
                a += 1;
                new &= new - 1;
            }
        }
        (r, a)
    }
}
