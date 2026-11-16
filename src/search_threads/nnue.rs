//! NNUE evaluation: a (768 x KING_BUCKETS -> HIDDEN) x 2 perspective network with SCReLU and
//! output buckets chosen by piece count. The accumulators are updated incrementally by the search.
//!
//! Each perspective sees the board from its own side (its back rank is rank 1). With more than
//! one king bucket, the board is also mirrored left-right when that side's king is on files e-h,
//! and the input weights are chosen by the king's (mirrored) square through `KING_BUCKET_LAYOUT`.

use crate::move_generation::Board;

/// Input weight sets, chosen by the perspective's own king square. 1 means no buckets and no
/// mirroring: the plain 768-input net.
pub const KING_BUCKETS: usize = 1;
const MIRROR: bool = KING_BUCKETS > 1;
/// Bucket of a king on the a-d files, by relative rank * 4 + file
const KING_BUCKET_LAYOUT: [u8; 32] = if KING_BUCKETS == 1 {
    [0; 32]
} else {
    [
        0, 1, 2, 3, // rank 1: a1, b1, c1, d1
        4, 4, 5, 5, // rank 2
        6, 6, 6, 6, // rank 3
        7, 7, 7, 7, // ranks 4-8
        7, 7, 7, 7, //
        7, 7, 7, 7, //
        7, 7, 7, 7, //
        7, 7, 7, 7, //
    ]
};
const _: () = assert!(KING_BUCKETS == 1 || KING_BUCKETS == 8);
pub const INPUTS: usize = 768 * KING_BUCKETS;
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

/// How one perspective sees the board: which input weight set, and how squares are flipped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct View {
    /// First input of the king bucket's weight set
    base: u16,
    /// XORed into squares: 56 flips black's view vertically, 7 mirrors left-right
    flip: u8,
}

impl View {
    /// The view of `perspective` (0 white, 1 black) whose king is on `king_square`.
    #[inline(always)]
    pub fn new(perspective: usize, king_square: usize) -> Self {
        let mut flip = if perspective == 0 { 0 } else { 56 };
        let relative = king_square ^ flip;
        if MIRROR && relative % 8 >= 4 {
            flip ^= 7;
        }
        // Mirroring leaves the king on files a-d, which is all the layout covers
        let king = relative ^ (flip & 7);
        let bucket = if MIRROR {
            KING_BUCKET_LAYOUT[(king / 8) * 4 + king % 8] as usize
        } else {
            0
        };
        Self {
            base: (bucket * 768) as u16,
            flip: flip as u8,
        }
    }

    /// Feature index of a piece (color 0 white) seen through this view of `perspective`.
    #[inline(always)]
    pub fn feature(self, perspective: usize, color: usize, piece: usize, square: usize) -> usize {
        self.base as usize
            + (color != perspective) as usize * 384
            + piece * 64
            + (square ^ self.flip as usize)
    }
}

/// Both perspectives' views of a board.
#[inline(always)]
pub fn views(board: &Board) -> [View; 2] {
    let white_king = (board.kings & board.white).trailing_zeros() as usize;
    let black_king = (board.kings & board.black).trailing_zeros() as usize;
    [View::new(0, white_king), View::new(1, black_king)]
}

#[inline(always)]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2)) as usize / 4).min(OUTPUT_BUCKETS - 1)
}

/// Both perspectives' hidden layer sums, and the views they were built with. White is 0.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub values: [[i16; HIDDEN]; 2],
    views: [View; 2],
}

impl Accumulator {
    pub fn from_board(board: &Board) -> Self {
        let net = network();
        let mut acc = Self {
            values: [net.feature_bias; 2],
            views: views(board),
        };
        for perspective in 0..2 {
            acc.refresh(net, board, perspective);
        }
        acc
    }

    /// Rebuilds one perspective from scratch.
    fn refresh(&mut self, net: &Network, board: &Board, perspective: usize) {
        let view = self.views[perspective];
        let out = &mut self.values[perspective];
        out.copy_from_slice(&net.feature_bias);
        for_each_piece(board, |color, piece, square| {
            let weights = &net.feature_weights[view.feature(perspective, color, piece, square)];
            for (v, w) in out.iter_mut().zip(weights) {
                *v += *w;
            }
        });
    }

    /// Moves from `parent` to the position now on `board`, one move later, given the pieces
    /// removed and added as (color, piece, square). A perspective whose king changed bucket or
    /// mirroring is rebuilt; the other is updated from the parent, reading it once and writing
    /// the child once.
    #[inline(always)]
    pub fn update(
        &mut self,
        parent: &Self,
        board: &Board,
        removed: &[(usize, usize, usize)],
        added: &[(usize, usize, usize)],
    ) {
        let net = network();
        self.views = views(board);
        for perspective in 0..2 {
            let view = self.views[perspective];
            if view != parent.views[perspective] {
                self.refresh(net, board, perspective);
                continue;
            }
            let row = |&(color, piece, square): &(usize, usize, usize)| {
                &net.feature_weights[view.feature(perspective, color, piece, square)]
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

/// Sum of clamp(v, 0, QA)^2 * w. Computed as (clamp(v) * w) * clamp(v): the first product fits
/// in an i16 because |w| <= 127 and v <= 255.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn screlu_dot(values: &[i16], weights: &[i16]) -> i32 {
    use std::arch::aarch64::*;
    const _: () = assert!(HIDDEN % 16 == 0);
    debug_assert!(values.len() == HIDDEN && weights.len() == HIDDEN);
    // SAFETY: NEON is always available on aarch64, and both slices hold HIDDEN elements, a
    // multiple of 16, so every load is in bounds. Each i32 lane sums HIDDEN / 16 products of at
    // most 255 * 255 * 127, which fits for HIDDEN up to several thousand.
    unsafe {
        let zero = vdupq_n_s16(0);
        let max = vdupq_n_s16(QA as i16);
        let mut acc = [vdupq_n_s32(0); 4];
        let (v_ptr, w_ptr) = (values.as_ptr(), weights.as_ptr());
        let mut i = 0;
        while i < HIDDEN {
            for half in 0..2 {
                let v = vld1q_s16(v_ptr.add(i + 8 * half));
                let w = vld1q_s16(w_ptr.add(i + 8 * half));
                let clamped = vminq_s16(vmaxq_s16(v, zero), max);
                let product = vmulq_s16(clamped, w);
                acc[2 * half] = vmlal_s16(acc[2 * half], vget_low_s16(product), vget_low_s16(clamped));
                acc[2 * half + 1] = vmlal_high_s16(acc[2 * half + 1], product, clamped);
            }
            i += 16;
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn screlu_dot(values: &[i16], weights: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    const _: () = assert!(HIDDEN % 16 == 0);
    debug_assert!(values.len() == HIDDEN && weights.len() == HIDDEN);
    // SAFETY: AVX2 is enabled at compile time (target-cpu=native), and both slices hold HIDDEN
    // elements, a multiple of 16, so every unaligned load is in bounds. madd sums adjacent pairs
    // into i32 lanes, which hold HIDDEN / 8 products of at most 255 * 255 * 127.
    unsafe {
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(QA as i16);
        let mut acc = _mm256_setzero_si256();
        let (v_ptr, w_ptr) = (values.as_ptr(), weights.as_ptr());
        let mut i = 0;
        while i < HIDDEN {
            let v = _mm256_loadu_si256(v_ptr.add(i) as *const __m256i);
            let w = _mm256_loadu_si256(w_ptr.add(i) as *const __m256i);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
            let product = _mm256_mullo_epi16(clamped, w);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(product, clamped));
            i += 16;
        }
        let sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b01));
        _mm_cvtsi128_si32(sum32)
    }
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn screlu_dot(values: &[i16], weights: &[i16]) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::{Accumulator, PieceBoards};
    use crate::move_generation::{self, Board};

    /// Incremental updates must match a rebuild from scratch after every move, including king
    /// moves that change bucket or cross the mirror line, castling, en passant and promotion.
    #[test]
    fn incremental_updates_match_a_full_refresh() {
        move_generation::move_lookup::initialize();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1pn1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];
        let mut checked = 0;
        for fen in starts {
            for _ in 0..40 {
                let mut board = Board::from_fen(fen);
                let mut acc = Accumulator::from_board(&board);
                for _ in 0..120 {
                    let moves = move_generation::legal_moves(&board);
                    if moves.length == 0 {
                        break;
                    }
                    let mov = moves.moves[(next() % moves.length as u64) as usize];
                    let before = PieceBoards::of(&board);
                    board.make_move(&mov);
                    let after = PieceBoards::of(&board);
                    let mut removed = [(0, 0, 0); 4];
                    let mut added = [(0, 0, 0); 4];
                    let (r, a) = before.diff(&after, &mut removed, &mut added);
                    let parent = acc;
                    acc.update(&parent, &board, &removed[..r], &added[..a]);
                    let fresh = Accumulator::from_board(&board);
                    assert_eq!(acc.values, fresh.values, "mismatch after {mov} in {board}");
                    assert_eq!(acc.views, fresh.views);
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000);
    }
}
