use crate::move_generation::mov::Move;
use crate::move_generation::{Board, move_lookup};
use crate::search_threads::eval_params::PARAMS;
use std::cmp::max;

/// A (middlegame, endgame) score pair, blended by game phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S(pub i32, pub i32);

// Indices into `PARAMS`. Keep in sync with `eval_params.rs` (and `tune.rs`, which prints it).
pub const MATERIAL: usize = 0;
pub const PSQT: usize = MATERIAL + 6;
pub const PASSED_PAWN: usize = PSQT + 6 * 64;
pub const KNIGHT_MOBILITY: usize = PASSED_PAWN + 8;
pub const BISHOP_MOBILITY: usize = KNIGHT_MOBILITY + 9;
pub const ROOK_MOBILITY: usize = BISHOP_MOBILITY + 14;
pub const QUEEN_MOBILITY: usize = ROOK_MOBILITY + 15;
pub const BISHOP_PAIR: usize = QUEEN_MOBILITY + 28;
pub const DOUBLED_PAWN: usize = BISHOP_PAIR + 1;
pub const ISOLATED_PAWN: usize = DOUBLED_PAWN + 1;
pub const ROOK_OPEN_FILE: usize = ISOLATED_PAWN + 1;
pub const ROOK_SEMI_OPEN_FILE: usize = ROOK_OPEN_FILE + 1;
pub const KING_SHIELD: usize = ROOK_SEMI_OPEN_FILE + 1;
pub const KING_ATTACK: usize = KING_SHIELD + 1;
pub const TEMPO: usize = KING_ATTACK + 4;
pub const NUM_PARAMS: usize = TEMPO + 1;

const _: () = assert!(NUM_PARAMS == PARAMS.len());

/// Full game phase: every minor is 1, rook 2, queen 4 (PeSTO's weights).
pub const MAX_PHASE: i32 = 24;

/// Receives how often each parameter applies (white minus black). Used by the tuner; the
/// search uses `NoTrace`, which compiles away.
pub trait Trace {
    fn add(&mut self, index: usize, count: i32);
}

pub struct NoTrace;

impl Trace for NoTrace {
    #[inline(always)]
    fn add(&mut self, _: usize, _: i32) {}
}

struct Accumulator<'a, T: Trace> {
    mg: i32,
    eg: i32,
    trace: &'a mut T,
}

impl<T: Trace> Accumulator<'_, T> {
    #[inline(always)]
    fn add(&mut self, index: usize, count: i32) {
        let weight = PARAMS[index];
        self.mg += weight.0 * count;
        self.eg += weight.1 * count;
        self.trace.add(index, count);
    }
}

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;

const fn file_mask(file: usize) -> u64 {
    FILE_A << file
}

const fn adjacent_files(file: usize) -> u64 {
    let mut mask = 0;
    if file > 0 {
        mask |= file_mask(file - 1);
    }
    if file < 7 {
        mask |= file_mask(file + 1);
    }
    mask
}

/// passed_pawn_masks\[color\]\[square\]: squares ahead on the same and adjacent files, where an
/// enemy pawn would stop the pawn from being passed. White is 0.
const PASSED_PAWN_MASKS: [[u64; 64]; 2] = {
    let mut masks = [[0; 64]; 2];
    let mut square = 0;
    while square < 64 {
        let file = square % 8;
        let rank = square / 8;
        let files = file_mask(file) | adjacent_files(file);
        let mut ahead_white = 0u64;
        let mut ahead_black = 0u64;
        let mut r = 0;
        while r < 8 {
            if r > rank {
                ahead_white |= 0xFF << (r * 8);
            }
            if r < rank {
                ahead_black |= 0xFF << (r * 8);
            }
            r += 1;
        }
        masks[0][square] = files & ahead_white;
        masks[1][square] = files & ahead_black;
        square += 1;
    }
    masks
};

#[inline(always)]
pub fn pawn_attacks(white: bool, pawns: u64) -> u64 {
    if white {
        ((pawns << 7) & !FILE_H) | ((pawns << 9) & !FILE_A)
    } else {
        ((pawns >> 9) & !FILE_H) | ((pawns >> 7) & !FILE_A)
    }
}

pub fn game_phase(board: &Board) -> i32 {
    let phase = (board.knights | board.bishops).count_ones() as i32
        + 2 * board.rooks.count_ones() as i32
        + 4 * board.queens.count_ones() as i32;
    phase.min(MAX_PHASE)
}

/// Static evaluation from white's point of view.
pub(crate) fn evaluate(board: &Board) -> i32 {
    evaluate_traced(board, &mut NoTrace)
}

pub fn evaluate_traced<T: Trace>(board: &Board, trace: &mut T) -> i32 {
    let mut acc = Accumulator {
        mg: 0,
        eg: 0,
        trace,
    };
    let occupied = board.all_pieces();
    let pieces = [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    ];

    for white in [true, false] {
        let sign = if white { 1 } else { -1 };
        let color = if white { 0 } else { 1 };
        let (us, them) = if white {
            (board.white, board.black)
        } else {
            (board.black, board.white)
        };
        let our_pawns = board.pawns & us;
        let their_pawns = board.pawns & them;
        let mobility_area = !us & !pawn_attacks(!white, their_pawns);
        let their_king = (board.kings & them).trailing_zeros() as u8;
        let king_zone = move_lookup::possible_king_moves(their_king) | (1 << their_king);
        let relative = |square: usize| if white { square } else { square ^ 56 };

        // Material and piece-square tables
        for (piece, &bitboard) in pieces.iter().enumerate() {
            let mut bits = bitboard & us;
            acc.add(MATERIAL + piece, sign * bits.count_ones() as i32);
            while bits != 0 {
                let square = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                acc.add(PSQT + piece * 64 + relative(square), sign);
            }
        }

        // Pawn structure
        for file in 0..8 {
            let on_file = (our_pawns & file_mask(file)).count_ones() as i32;
            if on_file > 1 {
                acc.add(DOUBLED_PAWN, sign * (on_file - 1));
            }
            if on_file > 0 && our_pawns & adjacent_files(file) == 0 {
                acc.add(ISOLATED_PAWN, sign * on_file);
            }
        }
        let mut pawns = our_pawns;
        while pawns != 0 {
            let square = pawns.trailing_zeros() as usize;
            pawns &= pawns - 1;
            if their_pawns & PASSED_PAWN_MASKS[color][square] == 0 {
                acc.add(PASSED_PAWN + relative(square) / 8, sign);
            }
        }

        if (board.bishops & us).count_ones() >= 2 {
            acc.add(BISHOP_PAIR, sign);
        }

        // Mobility and attacks on the enemy king's zone
        let mut knights = board.knights & us;
        while knights != 0 {
            let square = knights.trailing_zeros() as u8;
            knights &= knights - 1;
            let attacks = move_lookup::possible_knight_moves(square);
            acc.add(
                KNIGHT_MOBILITY + (attacks & mobility_area).count_ones() as usize,
                sign,
            );
            acc.add(
                KING_ATTACK,
                sign * (attacks & king_zone).count_ones() as i32,
            );
        }
        let mut bishops = board.bishops & us;
        while bishops != 0 {
            let square = bishops.trailing_zeros() as u8;
            bishops &= bishops - 1;
            let attacks = move_lookup::possible_bishop_moves(square, occupied);
            acc.add(
                BISHOP_MOBILITY + (attacks & mobility_area).count_ones() as usize,
                sign,
            );
            acc.add(
                KING_ATTACK + 1,
                sign * (attacks & king_zone).count_ones() as i32,
            );
        }
        let mut rooks = board.rooks & us;
        while rooks != 0 {
            let square = rooks.trailing_zeros() as u8;
            rooks &= rooks - 1;
            let attacks = move_lookup::possible_rook_moves(square, occupied);
            acc.add(
                ROOK_MOBILITY + (attacks & mobility_area).count_ones() as usize,
                sign,
            );
            acc.add(
                KING_ATTACK + 2,
                sign * (attacks & king_zone).count_ones() as i32,
            );
            let file = file_mask(square as usize % 8);
            if our_pawns & file == 0 {
                if their_pawns & file == 0 {
                    acc.add(ROOK_OPEN_FILE, sign);
                } else {
                    acc.add(ROOK_SEMI_OPEN_FILE, sign);
                }
            }
        }
        let mut queens = board.queens & us;
        while queens != 0 {
            let square = queens.trailing_zeros() as u8;
            queens &= queens - 1;
            let attacks = move_lookup::possible_queen_moves(square, occupied);
            acc.add(
                QUEEN_MOBILITY + (attacks & mobility_area).count_ones() as usize,
                sign,
            );
            acc.add(
                KING_ATTACK + 3,
                sign * (attacks & king_zone).count_ones() as i32,
            );
        }

        // Pawn shield in front of a king that is still on its back two ranks
        let king = (board.kings & us).trailing_zeros() as usize;
        let king_rank = relative(king) / 8;
        if king_rank <= 1 {
            let files = file_mask(king % 8) | adjacent_files(king % 8);
            let ranks = if white {
                0xFFFFu64 << ((king / 8 + 1) * 8)
            } else {
                0xFFFFu64 << ((king / 8).saturating_sub(2) * 8)
            };
            acc.add(
                KING_SHIELD,
                sign * (our_pawns & files & ranks).count_ones() as i32,
            );
        }
    }

    acc.add(TEMPO, if board.white_to_move() { 1 } else { -1 });

    let phase = game_phase(board);
    (acc.mg * phase + acc.eg * (MAX_PHASE - phase)) / MAX_PHASE
}

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

pub fn mvv_lva(attacker: usize, victim: usize) -> i32 {
    PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker]
}

pub fn static_exchange_evaluation(board: &Board, mov: &Move) -> i32 {
    let mut gain = [0; 32];
    let mut depth = 0;

    let mut white = !board.white_to_move();
    let mut occupied_squares = board.white | board.black;

    let capture_type = if mov.flags() == 0b0001 {
        0
    } else {
        match board.get_piece_at_square(mov.ending_index()) {
            Some(piece_type) => piece_type as usize,
            None => usize::MAX,
        }
    };
    // Quiet moves start from nothing captured, so SEE says whether the square is safe
    gain[0] = if capture_type == usize::MAX {
        0
    } else {
        PIECE_VALUES[capture_type]
    };

    let mut current_attacker_type =
        board.get_piece_at_square(mov.starting_index()).unwrap() as usize;
    if current_attacker_type == 0 {
        current_attacker_type = match mov.flags() {
            0b0100 => 4,
            0b0101 => 3,
            0b0110 => 2,
            0b0111 => 1,
            _ => 0,
        };
    }

    // The moving piece leaves its origin, and an en-passant capture also
    // removes the pawn that is not on the destination square.
    occupied_squares &= !mov.starting_mask();
    if mov.flags() == 0b0001 {
        let captured_pawn_square = if mov.ending_index() % 8 < mov.starting_index() % 8 {
            mov.starting_index() - 1
        } else {
            mov.starting_index() + 1
        };
        occupied_squares &= !(1 << captured_pawn_square);
    }
    let mut attackers = get_attackers(board, mov.ending_index(), occupied_squares);

    loop {
        let friendly_pieces = if white { board.white } else { board.black };
        let active_attackers = friendly_pieces & attackers & occupied_squares;

        // Find the least valuable attacker available for this side
        let mut chosen_square = None;
        let mut selected_attacker_type = None;
        for piece_type in 0..6 {
            let piece_mask = match piece_type {
                0 => board.pawns,
                1 => board.knights,
                2 => board.bishops,
                3 => board.rooks,
                4 => board.queens,
                5 => board.kings,
                _ => u64::MAX,
            };

            let subset = active_attackers & piece_mask;
            if subset != 0 {
                selected_attacker_type = Some(piece_type);
                chosen_square = Some(subset.trailing_zeros() as u8);
                break;
            }
        }

        // If no captures to be made, break
        let attacker_sq = match chosen_square {
            Some(sq) => sq,
            None => break,
        };
        let selected_attacker_type = selected_attacker_type.unwrap();

        depth += 1;
        gain[depth] = PIECE_VALUES[current_attacker_type] - gain[depth - 1];
        if max(-gain[depth - 1], gain[depth]) < 0 {
            break;
        }

        // Remove the capturing piece from the virtual board state
        occupied_squares &= !(1 << attacker_sq);

        // Re-trigger Bishop/Queen rays through the newly vacant square
        attackers |= move_lookup::possible_bishop_moves(mov.ending_index(), occupied_squares)
            & (board.bishops | board.queens);
        // Re-trigger Rook/Queen rays through the newly vacant square
        attackers |= move_lookup::possible_rook_moves(mov.ending_index(), occupied_squares)
            & (board.rooks | board.queens);

        // If king capture
        if selected_attacker_type == 5 {
            let enemy_pieces = if white { board.black } else { board.white };
            // And enemy pieces can attack the king post-move
            if attackers & enemy_pieces & occupied_squares != 0 {
                depth -= 1; // Roll back this move, we're putting ourselves in check
            }
            break; // Break either way since king capture is always a last resort
        }

        current_attacker_type = selected_attacker_type;
        if current_attacker_type == 0
            && ((white && mov.ending_index() / 8 == 7) || (!white && mov.ending_index() / 8 == 0))
        {
            current_attacker_type = 4;
        }
        white = !white;
    }

    while depth > 0 {
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth]);
        depth -= 1;
    }

    gain[0]
}

fn get_attackers(board: &Board, square: u8, occupied_squares: u64) -> u64 {
    let mut attackers = 0;

    if square / 8 > 0 {
        if square % 8 != 0 {
            attackers |= (1 << (square - 9)) & board.pawns & board.white;
        }
        if square % 8 != 7 {
            attackers |= (1 << (square - 7)) & board.pawns & board.white;
        }
    }
    if square / 8 < 7 {
        if square % 8 != 0 {
            attackers |= (1 << (square + 7)) & board.pawns & board.black;
        }
        if square % 8 != 7 {
            attackers |= (1 << (square + 9)) & board.pawns & board.black;
        }
    }

    attackers |= move_lookup::possible_knight_moves(square) & board.knights;
    attackers |= move_lookup::possible_bishop_moves(square, occupied_squares) & board.bishops;
    attackers |= move_lookup::possible_rook_moves(square, occupied_squares) & board.rooks;
    attackers |= move_lookup::possible_queen_moves(square, occupied_squares) & board.queens;
    attackers |= move_lookup::possible_king_moves(square) & board.kings;
    attackers
}

#[cfg(test)]
mod tests {
    use crate::move_generation::Board;
    use crate::move_generation::mov::Move;
    use crate::search_threads::evaluation::static_exchange_evaluation;

    fn see(fen: &str, starting_square: u8, ending_square: u8) -> i32 {
        let board = Board::from_fen(fen);
        static_exchange_evaluation(&board, &Move::regular(starting_square, ending_square))
    }

    #[test]
    fn see_values_capture_without_recapture() {
        assert_eq!(see("4k3/8/8/8/3r4/2B5/8/4K3 w - - 0 1", 18, 27), 500);
    }

    #[test]
    fn see_accounts_for_a_recapture() {
        assert_eq!(see("4k3/8/8/4p3/3r4/2B5/8/4K3 w - - 0 1", 18, 27), 170);
    }

    #[test]
    fn see_uses_the_correct_pawn_attack_direction() {
        assert_eq!(see("4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1", 28, 35), 500);
        assert_eq!(see("4k3/8/8/4p3/3R4/8/8/4K3 b - - 0 1", 36, 27), 500);
    }
}
