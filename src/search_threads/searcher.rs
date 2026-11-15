use crate::move_generation;
use crate::move_generation::mov::Move;
use crate::move_generation::mov::piece_type::PAWN;
use crate::move_generation::{Board, Moves, MovesWithMetadata};
use crate::search_threads::transposition_table::{EntryResult, TranspositionTable};
use crate::search_threads::{SearchResult, evaluation, print_info};
use minstant::Instant;
use rand::seq::SliceRandom;
use std::cmp::{Reverse, max};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAX_DEPTH: u8 = 16;

pub struct Search {
    pub(crate) nodes_searched: u64,
    killer_moves: [Option<Move>; MAX_DEPTH as usize],
}

impl Search {
    pub fn new() -> Self {
        Self {
            nodes_searched: 0,
            killer_moves: [None; MAX_DEPTH as usize],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search(
        &mut self,
        board: &mut Board,
        transposition_table: &TranspositionTable,
        stop_signal: &AtomicBool,
        time_left_ms: u64,
        increment: u64,
        nodes_searched: &AtomicU64,
        search_id: usize,
    ) -> SearchResult {
        self.nodes_searched = 0;
        self.killer_moves.fill(None);
        let legal_moves = move_generation::legal_moves(board);
        if legal_moves.length == 0 {
            panic!(
                "Attempted to search_threads and eval board {} when we're checkmated",
                board
            );
        }
        if legal_moves.length == 1 {
            return SearchResult {
                depth: 0,
                best_ply: 0,
                score: 0,
                best_move: legal_moves.moves[0],
            };
        }

        let color = if board.white_to_move() { 1 } else { -1 };
        let transposition_table_result = transposition_table.get(board.zobrist_hash);
        let mut tt_move = None;
        if let Some(result) = transposition_table_result {
            tt_move = Some(result.best_move);
        }
        let mut sorted_moves = self.sort_moves(board, &legal_moves, tt_move, None, color);
        if search_id > 0 {
            sorted_moves[1..].shuffle(&mut rand::rng());
        }
        let mut sorted_moves: Vec<(Move, i32)> = sorted_moves.iter().map(|&m| (m, 0)).collect();

        let mut best_score = -i32::MAX;
        let mut best_move = sorted_moves[0].0;
        let mut best_depth = 0;
        let mut best_ply = 0;
        let timer = Instant::now();
        let mut last_print_time = timer.elapsed().as_secs_f64();
        let time_ok = || {
            let elapsed = timer.elapsed().as_millis() as u64;
            !stop_signal.load(Relaxed)
                && elapsed < (time_left_ms / 20 + increment / 2)
                && elapsed < 10000
        };
        'iterative_deepening: for depth in 0..MAX_DEPTH {
            if depth > 0 {
                sorted_moves.sort_by_key(|&(_, eval)| Reverse(eval));
                if search_id > 0 {
                    sorted_moves[1..].shuffle(&mut rand::rng());
                }
            }

            let mut depth_best_score = -i32::MAX;
            let mut depth_best_move = best_move;
            let mut depth_best_ply = 0;
            for i in 0..sorted_moves.len() {
                let (mov, sort_eval) = &mut sorted_moves[i];
                let metadata = board.make_move(mov);

                let result = if i == 0 || depth < 2 {
                    self.negamax(
                        board,
                        transposition_table,
                        depth,
                        0,
                        -i32::MAX,
                        -depth_best_score,
                        -color,
                        &time_ok,
                        nodes_searched,
                    )
                } else {
                    // Moves that aren't the first one probably aren't as good, so do a null window search
                    // to confirm that they're not better than the best move
                    let pvs = self.negamax(
                        board,
                        transposition_table,
                        depth,
                        0,
                        -depth_best_score - 1,
                        -depth_best_score,
                        -color,
                        &time_ok,
                        nodes_searched,
                    );
                    if let Some(pvs) = pvs {
                        // If the move's not worth looking at, just return that search result
                        let value = -pvs.0;
                        if value <= depth_best_score {
                            Some(pvs)
                        } else {
                            // If it turns out it is actually better, we need a full search to quantify how much better
                            self.negamax(
                                board,
                                transposition_table,
                                depth,
                                0,
                                -i32::MAX,
                                -depth_best_score,
                                -color,
                                &time_ok,
                                nodes_searched,
                            )
                        }
                    } else {
                        None
                    }
                };

                if result.is_none() {
                    board.unmake_move(mov, metadata);
                    break 'iterative_deepening;
                }

                let (eval, searched_ply) = result.unwrap();
                let eval = -eval;

                *sort_eval = eval;

                board.unmake_move(mov, metadata);
                if eval > depth_best_score {
                    depth_best_score = eval;
                    depth_best_move = *mov;
                    depth_best_ply = searched_ply;
                }

                if search_id == 0 {
                    let score = if depth > 0 { best_score } else { 0 };
                    let time = timer.elapsed().as_secs_f64();
                    let nodes =
                        nodes_searched.load(Ordering::Relaxed) + (self.nodes_searched & 2047);
                    if timer.elapsed().as_secs_f64() - last_print_time > 0.05 {
                        print_info(depth, depth_best_ply, score, time, nodes);
                        last_print_time = timer.elapsed().as_secs_f64();
                    }
                }

                if !time_ok() {
                    break 'iterative_deepening;
                }
            }
            best_score = depth_best_score;
            best_move = depth_best_move;
            best_depth = depth;
            best_ply = depth_best_ply;
            // We've reached the maximum depth, any further search is meaningless so exit early
            if depth + 1 == MAX_DEPTH {
                stop_signal.store(true, Ordering::Relaxed);
                break 'iterative_deepening;
            }
        }

        nodes_searched.fetch_add(self.nodes_searched % 2047, Ordering::Relaxed);
        SearchResult {
            depth: best_depth,
            best_ply,
            score: best_score,
            best_move,
        }
    }

    pub(super) const IMMEDIATE_MATE_SCORE: i32 = i32::MAX / 2;

    #[allow(clippy::too_many_arguments)]
    fn negamax<F>(
        &mut self,
        board: &mut Board,
        transposition_table: &TranspositionTable,
        depth: u8,
        search_ply: u8,
        alpha: i32,
        beta: i32,
        color: i32,
        time_ok: &F,
        nodes_searched: &AtomicU64,
    ) -> Option<(i32, u8)>
    where
        F: Fn() -> bool,
    {
        if board.get_halfmove_clock() > 100 || board.times_played() >= 3 {
            return Some((0, search_ply));
        }
        self.count_nodes(nodes_searched);
        let mut evaluation_type = 1;

        // Alpha can't possibly be worse than getting checkmated right now, if it is it means we already found a
        // faster mate elsewhere
        // Same for beta
        let mut alpha = alpha.max(-Self::IMMEDIATE_MATE_SCORE + search_ply as i32);
        let mut beta = beta.min(Self::IMMEDIATE_MATE_SCORE - search_ply as i32);
        if alpha >= beta {
            return Some((alpha, search_ply));
        }

        let moves = move_generation::legal_moves_with_metadata(board);
        if moves.length == 0 {
            return if moves.check {
                Some((search_ply as i32 - Self::IMMEDIATE_MATE_SCORE, search_ply)) // Checkmate
            } else {
                Some((0, search_ply)) // Draw by no legal moves
            };
        }

        let tt_result = transposition_table.get(board.zobrist_hash);
        let mut tt_move = None;
        if let Some(tt_result) = tt_result {
            tt_move = Some(tt_result.best_move);
            if tt_result.depth >= depth {
                match tt_result.node_type {
                    1 => beta = beta.min(tt_result.score),           // Upper bound
                    0 => return Some((tt_result.score, search_ply)), // Exact
                    -1 => alpha = alpha.max(tt_result.score),        // Lower bound
                    _ => {}
                }
                if alpha >= beta {
                    return Some((tt_result.score, search_ply));
                }
            }
        }

        if depth == 0 || search_ply >= MAX_DEPTH {
            return self.quiescence(
                board,
                moves,
                alpha,
                beta,
                search_ply + 1,
                color,
                time_ok,
                nodes_searched,
            );
        }

        let mut extensions = -1;
        if moves.length == 1 {
            extensions += 1;
        }
        if moves.check {
            extensions += 1;
        }

        let moves = moves.strip_metadata();
        let sorted_moves = self.sort_moves(
            board,
            &moves,
            tt_move,
            self.killer_moves[search_ply as usize],
            color,
        );

        let mut best_move = sorted_moves[0];
        let mut best_value = -i32::MAX;
        let mut best_searched_ply = search_ply;
        for i in 0..sorted_moves.len() {
            if !time_ok() {
                return None;
            }
            let mov = sorted_moves[i];

            let metadata = board.make_move(&mov);

            let search = if i == 0 || depth < 2 {
                self.negamax(
                    board,
                    transposition_table,
                    max(0, depth as i8 + extensions) as u8,
                    search_ply + 1,
                    -beta,
                    -alpha,
                    -color,
                    time_ok,
                    nodes_searched,
                )
            } else {
                let full_depth = max(0, depth as i8 + extensions) as u8;
                let reduced = depth > 2 && sorted_moves.len() > 4 && i > 2;
                let search_depth = if reduced {
                    max(0, depth as i8 + extensions - 1) as u8
                } else {
                    full_depth
                };
                // Moves that aren't first probably aren't worth looking at, so do a null-window
                // search (at reduced depth for late moves) to confirm that's the case
                let mut result = self.negamax(
                    board,
                    transposition_table,
                    search_depth,
                    search_ply + 1,
                    -alpha - 1,
                    -alpha,
                    -color,
                    time_ok,
                    nodes_searched,
                );
                // A reduced search beat alpha: verify at full depth, still with a null window
                if reduced && result.is_some_and(|r| -r.0 > alpha) {
                    result = self.negamax(
                        board,
                        transposition_table,
                        full_depth,
                        search_ply + 1,
                        -alpha - 1,
                        -alpha,
                        -color,
                        time_ok,
                        nodes_searched,
                    );
                }
                // Only a move that lands strictly inside (alpha, beta) needs an exact score.
                // In null-window nodes beta == alpha + 1, so this never re-searches there.
                if result.is_some_and(|r| -r.0 > alpha && -r.0 < beta) {
                    result = self.negamax(
                        board,
                        transposition_table,
                        full_depth,
                        search_ply + 1,
                        -beta,
                        -alpha,
                        -color,
                        time_ok,
                        nodes_searched,
                    );
                }
                result
            };

            if search.is_none() {
                board.unmake_move(&mov, metadata);
                return None;
            }

            let (value, searched_ply) = search.unwrap();

            let value = -value;

            board.unmake_move(&mov, metadata);
            if value > best_value {
                best_value = value;
                best_searched_ply = searched_ply;
            }
            if value > alpha {
                alpha = value;
                // Only trust moves that actually raised alpha. In fail-low nodes every child score
                // is just an upper bound, so storing those in TT poisons the move ordering.
                best_move = mov;
                evaluation_type = 0;
                if alpha >= beta {
                    evaluation_type = -1;
                    self.killer_moves[search_ply as usize] = Some(mov);
                    break;
                }
            }
        }

        let tt_score = if evaluation_type == -1 { beta } else { alpha };
        transposition_table.insert(
            board.zobrist_hash,
            EntryResult::new(depth, tt_score, evaluation_type, best_move),
        );
        Some((best_value, best_searched_ply))
    }

    #[allow(clippy::too_many_arguments)]
    fn quiescence<F>(
        &mut self,
        board: &mut Board,
        moves: MovesWithMetadata,
        alpha: i32,
        beta: i32,
        search_ply: u8,
        color: i32,
        time_ok: &F,
        nodes_searched: &AtomicU64,
    ) -> Option<(i32, u8)>
    where
        F: Fn() -> bool,
    {
        if !time_ok() {
            return None;
        }

        if board.get_halfmove_clock() > 100 || board.times_played() >= 3 {
            return Some((0, search_ply));
        }

        self.count_nodes(nodes_searched);

        if moves.length == 0 {
            return if moves.check {
                Some((search_ply as i32 - Self::IMMEDIATE_MATE_SCORE, search_ply)) // Checkmate
            } else {
                Some((0, search_ply)) // Draw by no legal moves
            };
        }

        let static_eval = evaluation::evaluate(board) * color;
        if static_eval >= beta {
            return Some((static_eval, search_ply));
        }
        let mut alpha = alpha;
        if static_eval > alpha {
            alpha = static_eval;
        }
        let mut best_value = static_eval;

        let moves = moves.filter_captures();
        if moves.length == 0 {
            return Some((best_value, search_ply));
        }
        let moves = self.sort_moves(board, &moves, None, None, color);

        let mut best_search_ply = search_ply;
        for mov in moves {
            if !time_ok() {
                return None;
            }

            let see = evaluation::static_exchange_evaluation(board, &mov);
            if see < 0 {
                continue;
            }

            let metadata = board.make_move(&mov);
            let result = self.quiescence(
                board,
                move_generation::legal_moves_with_metadata(board),
                -beta,
                -alpha,
                search_ply + 1,
                -color,
                time_ok,
                nodes_searched,
            );
            board.unmake_move(&mov, metadata);
            result?;

            let (eval, searched_ply) = result.unwrap();
            let eval = -eval;

            if eval > best_value {
                best_value = eval;
                best_search_ply = searched_ply;
            }
            if eval > alpha {
                alpha = eval;
                if alpha >= beta {
                    break;
                }
            }
        }

        Some((best_value, best_search_ply))
    }

    fn sort_moves(
        &self,
        board: &mut Board,
        moves: &Moves,
        tt_move: Option<Move>,
        killer_move: Option<Move>,
        color: i32,
    ) -> Vec<Move> {
        let mut sorted_moves: Vec<(&Move, i32)> = Vec::with_capacity(moves.length);
        for i in 0..moves.length {
            let mov = &moves.moves[i];
            let value;
            if let Some(tt_move) = tt_move
                && tt_move.eq(mov)
            {
                value = 1 << 20;
            } else if let Some(killer_move) = killer_move
                && killer_move.eq(mov)
            {
                if mov.flags() != 0b0001 && board.get_piece_at_square(mov.ending_index()).is_none()
                {
                    value = 10_000; // If killer move is quiet, score below captures but above normal quiets
                } else {
                    value = 1 << 19; // If capture, score below TT best move but above everything else
                }
            } else {
                let capture_type = if mov.flags() == 0b0001 {
                    Some(PAWN)
                } else {
                    board.get_piece_at_square(mov.ending_index())
                };

                value = if let Some(capture_type) = capture_type {
                    let eval = evaluation::static_exchange_evaluation(board, mov);
                    if eval != 0 {
                        eval + (100_000 * eval.signum()) // Bad captures are scored below quiet moves
                    } else {
                        30_000
                            + evaluation::mvv_lva(
                                board.get_piece_at_square(mov.starting_index()).unwrap() as usize,
                                capture_type as usize,
                            )
                    }
                } else {
                    let metadata = board.make_move(mov);
                    let eval = evaluation::evaluate(board) * color;
                    board.unmake_move(mov, metadata);
                    eval
                };
            }
            sorted_moves.push((mov, value));
        }
        sorted_moves.sort_by_key(|mov| Reverse(mov.1));
        sorted_moves.iter().map(|mov| *mov.0).collect()
    }

    fn count_nodes(&mut self, nodes_searched: &AtomicU64) {
        self.nodes_searched += 1;
        if self.nodes_searched % 2047 == 0 {
            nodes_searched.fetch_add(2048, Ordering::Relaxed);
        }
    }
}
