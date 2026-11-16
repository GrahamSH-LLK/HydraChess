use crate::move_generation;
use crate::move_generation::mov::Move;
use crate::move_generation::mov::piece_type::{KING, PAWN};
use crate::move_generation::{Board, MovesWithMetadata};
use crate::search_threads::nnue::{Accumulator, PieceBoards};
use crate::search_threads::transposition_table::{EntryResult, TranspositionTable};
use crate::search_threads::{SearchResult, evaluation};
use minstant::Instant;
use std::io::Write;
use std::io::stdout;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};

pub const MAX_DEPTH: u8 = 64;
const MAX_PLY: usize = 128;

pub const MATE: i32 = 32_000;
const INFINITY: i32 = MATE + 1;
/// Scores beyond this are mate scores
const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

// Transposition table node types
const EXACT: i8 = 0;
const UPPER_BOUND: i8 = 1;
const LOWER_BOUND: i8 = -1;

const MAX_HISTORY: i32 = 16_384;

/// How often (in nodes) to check the clock and stop flags
const CHECK_INTERVAL: u64 = 1024;

/// Marks a ply that was reached by a null move (or the root), which has no piece or square
const NO_PIECE: usize = usize::MAX;
/// Pieces are keyed by type and color: white pawn 0 .. black king 11
const PIECE_KEYS: usize = 12;
const CONTINUATION_SIZE: usize = PIECE_KEYS * 64;

/// What the GUI asked for in `go`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Limits {
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: u64,
    pub black_increment_ms: u64,
    /// The side to move's clock, filled in from the two above when the search starts
    pub time_left_ms: Option<u64>,
    pub increment_ms: u64,
    pub movetime_ms: Option<u64>,
    pub depth: Option<u8>,
    pub nodes: Option<u64>,
    /// Don't start another iteration past this many nodes (this thread's own count)
    pub soft_nodes: Option<u64>,
    pub infinite: bool,
}

/// State shared by all search threads.
pub struct SearchContext<'a> {
    pub transposition_table: &'a TranspositionTable,
    /// Set by the GUI's `stop`
    pub stop_signal: &'a AtomicBool,
    /// Set by the main thread when it finishes, to stop the helpers
    pub abort_signal: &'a AtomicBool,
    pub nodes_searched: &'a AtomicU64,
}

/// The move that led to a ply, for counter-move and continuation history lookups.
#[derive(Clone, Copy)]
struct PlyMove {
    /// `NO_PIECE` after a null move or at the root
    piece: usize,
    to: usize,
}

impl PlyMove {
    const NONE: Self = Self {
        piece: NO_PIECE,
        to: 0,
    };

    fn key(self) -> Option<usize> {
        (self.piece != NO_PIECE).then(|| self.piece * 64 + self.to)
    }
}

pub struct Search {
    id: usize,
    nodes: u64,
    flushed_nodes: u64,
    stopped: bool,
    timer: Instant,
    soft_limit_ms: u64,
    hard_limit_ms: u64,
    max_nodes: u64,
    root_depth: i32,
    killer_moves: [[Move; 2]; MAX_PLY + 1],
    /// history\[side to move\]\[from\]\[to\]
    history: Box<[[[i32; 64]; 64]; 2]>,
    /// The quiet move that refuted the previous move, by the previous move's piece and square
    counter_moves: Box<[[Move; 64]; PIECE_KEYS]>,
    /// Quiet move score by (move 1 or 2 plies ago, this move), flattened
    continuation_history: Vec<i32>,
    /// capture_history\[piece key\]\[to\]\[captured piece type\]
    capture_history: Box<[[[i32; 6]; 64]; PIECE_KEYS]>,
    principal_variation: Box<[[Move; MAX_PLY]; MAX_PLY]>,
    principal_variation_length: [usize; MAX_PLY],
    late_move_reductions: Box<[[i32; 64]; 64]>,
    /// Static eval per ply, to tell whether the position is improving
    eval_stack: [i32; MAX_PLY + 1],
    move_stack: [PlyMove; MAX_PLY + 1],
    /// The move a singular extension search leaves out, per ply
    excluded_moves: [Move; MAX_PLY + 1],
    root_best_move: Option<Move>,
    /// NNUE accumulators per ply
    accumulators: Vec<Accumulator>,
    /// Whether to print `info` lines (main thread only)
    pub print_info: bool,
}

impl Search {
    pub fn new(id: usize) -> Self {
        let mut late_move_reductions = Box::new([[0; 64]; 64]);
        for depth in 1..64 {
            for moves in 1..64 {
                late_move_reductions[depth][moves] =
                    (0.75 + (depth as f64).ln() * (moves as f64).ln() / 2.25) as i32;
            }
        }
        Self {
            id,
            nodes: 0,
            flushed_nodes: 0,
            stopped: false,
            timer: Instant::now(),
            soft_limit_ms: u64::MAX,
            hard_limit_ms: u64::MAX,
            max_nodes: u64::MAX,
            root_depth: 0,
            killer_moves: [[Move::null(); 2]; MAX_PLY + 1],
            history: Box::new([[[0; 64]; 64]; 2]),
            counter_moves: Box::new([[Move::null(); 64]; PIECE_KEYS]),
            continuation_history: vec![0; CONTINUATION_SIZE * CONTINUATION_SIZE],
            capture_history: Box::new([[[0; 6]; 64]; PIECE_KEYS]),
            principal_variation: Box::new([[Move::null(); MAX_PLY]; MAX_PLY]),
            principal_variation_length: [0; MAX_PLY],
            late_move_reductions,
            eval_stack: [0; MAX_PLY + 1],
            move_stack: [PlyMove::NONE; MAX_PLY + 1],
            excluded_moves: [Move::null(); MAX_PLY + 1],
            root_best_move: None,
            accumulators: vec![Accumulator::from_board(&Board::starting()); MAX_PLY + 2],
            print_info: id == 0,
        }
    }

    pub fn new_game(&mut self) {
        self.history
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|h| *h = 0);
        self.counter_moves
            .iter_mut()
            .flatten()
            .for_each(|m| *m = Move::null());
        self.continuation_history.iter_mut().for_each(|h| *h = 0);
        self.capture_history
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|h| *h = 0);
    }

    fn is_main(&self) -> bool {
        self.id == 0
    }

    /// Converts the GUI's clock into a soft limit (don't start another iteration) and a hard
    /// limit (abort the search).
    fn set_time_limits(&mut self, limits: &Limits) {
        self.soft_limit_ms = u64::MAX;
        self.hard_limit_ms = u64::MAX;
        self.max_nodes = limits.nodes.unwrap_or(u64::MAX);
        // Only the main thread manages time; helpers stop when it does
        if !self.is_main() || limits.infinite {
            return;
        }
        const MOVE_OVERHEAD_MS: u64 = 20;
        if let Some(movetime) = limits.movetime_ms {
            let limit = movetime.saturating_sub(MOVE_OVERHEAD_MS).max(1);
            self.soft_limit_ms = limit;
            self.hard_limit_ms = limit;
        } else if let Some(time_left) = limits.time_left_ms {
            let available = time_left.saturating_sub(MOVE_OVERHEAD_MS).max(1);
            let soft = available / 40 + limits.increment_ms * 3 / 4;
            let hard = (available / 8 + limits.increment_ms).max(soft);
            self.soft_limit_ms = soft.min(available).max(1);
            self.hard_limit_ms = hard.min(available).max(1);
        }
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        context: &SearchContext,
        limits: &Limits,
    ) -> SearchResult {
        self.timer = Instant::now();
        self.nodes = 0;
        self.flushed_nodes = 0;
        self.stopped = false;
        self.root_best_move = None;
        self.killer_moves = [[Move::null(); 2]; MAX_PLY + 1];
        self.move_stack = [PlyMove::NONE; MAX_PLY + 1];
        self.excluded_moves = [Move::null(); MAX_PLY + 1];
        // Keep some history between moves, but let it adapt to the new position
        self.history
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|h| *h /= 2);
        self.continuation_history.iter_mut().for_each(|h| *h /= 2);
        self.capture_history
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|h| *h /= 2);
        self.set_time_limits(limits);
        self.accumulators[0] = Accumulator::from_board(board);

        let legal_moves = move_generation::legal_moves(board);
        if legal_moves.length == 0 {
            return SearchResult {
                depth: 0,
                score: 0,
                best_move: Move::null(),
            };
        }
        let mut result = SearchResult {
            depth: 0,
            score: 0,
            best_move: legal_moves.moves[0],
        };
        if legal_moves.length == 1 && self.is_main() && !limits.infinite {
            return result;
        }

        let max_depth = limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        // Half of the helper threads start a ply deeper, so the threads spread across depths
        let start_depth = if self.id % 2 == 1 {
            2.min(max_depth)
        } else {
            1
        };
        let mut score = 0;
        let mut previous_score = 0;
        let mut stability = 0usize;
        for depth in start_depth..=max_depth {
            self.root_depth = depth as i32;
            score = self.aspiration_search(board, context, depth as i32, score);
            if self.stopped {
                // An interrupted iteration is only trusted if it already found a new best move
                // (the previous best is always searched first)
                if let Some(best) = self.root_best_move
                    && best != result.best_move
                    && self.principal_variation_length[0] > 0
                {
                    result.best_move = best;
                }
                break;
            }
            let best_move = self.principal_variation[0][0];
            if best_move == result.best_move {
                stability += 1;
            } else {
                stability = 0;
            }
            result = SearchResult {
                depth,
                score,
                best_move,
            };
            self.root_best_move = Some(result.best_move);
            if self.is_main() {
                if self.print_info {
                    self.print_info(depth, score, context);
                }
                // Spend less time when the best move keeps holding, more when it flips or
                // the score drops
                const STABILITY_SCALE: [f64; 7] = [1.8, 1.4, 1.15, 1.0, 0.9, 0.8, 0.75];
                let mut scale = STABILITY_SCALE[stability.min(6)];
                if depth > 1 && score < previous_score - 30 {
                    scale *= 1.25;
                }
                let soft_limit = (self.soft_limit_ms as f64 * scale) as u64;
                if self.elapsed_ms() >= soft_limit.min(self.hard_limit_ms) {
                    break;
                }
            }
            previous_score = score;
            if limits.soft_nodes.is_some_and(|soft| self.nodes >= soft) {
                break;
            }
        }
        self.flush_nodes(context);
        result
    }

    /// Searches the root with a narrow window around the previous score, widening on failure.
    fn aspiration_search(
        &mut self,
        board: &mut Board,
        context: &SearchContext,
        depth: i32,
        previous_score: i32,
    ) -> i32 {
        let mut delta = 25;
        let (mut alpha, mut beta) = if depth >= 5 && previous_score.abs() < MATE_BOUND {
            (previous_score - delta, previous_score + delta)
        } else {
            (-INFINITY, INFINITY)
        };
        loop {
            let score = self.negamax(board, context, depth, 0, alpha, beta, false);
            if self.stopped {
                return score;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INFINITY);
            } else if score >= beta {
                beta = (score + delta).min(INFINITY);
            } else {
                return score;
            }
            delta += delta / 2;
            if delta > 1000 {
                alpha = -INFINITY;
                beta = INFINITY;
            }
        }
    }

    fn print_info(&self, depth: u8, score: i32, context: &SearchContext) {
        let elapsed_ms = self.elapsed_ms();
        let nodes = context.nodes_searched.load(Relaxed) + (self.nodes - self.flushed_nodes);
        let nps = nodes * 1000 / elapsed_ms.max(1);
        let score = if score >= MATE_BOUND {
            format!("mate {}", (MATE - score + 1) / 2)
        } else if score <= -MATE_BOUND {
            format!("mate {}", -(MATE + score) / 2)
        } else {
            format!("cp {}", score)
        };
        let principal_variation = self.principal_variation[0][..self.principal_variation_length[0]]
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let mut out = stdout().lock();
        writeln!(
            out,
            "info depth {depth} score {score} nodes {nodes} nps {nps} time {elapsed_ms} pv {principal_variation}"
        )
        .unwrap();
        out.flush().unwrap();
    }

    fn elapsed_ms(&self) -> u64 {
        self.timer.elapsed().as_millis() as u64
    }

    fn flush_nodes(&mut self, context: &SearchContext) {
        context
            .nodes_searched
            .fetch_add(self.nodes - self.flushed_nodes, Relaxed);
        self.flushed_nodes = self.nodes;
    }

    /// Counts a node, and every few nodes checks whether the search has to stop.
    fn count_node(&mut self, context: &SearchContext) {
        self.nodes += 1;
        if self.nodes % CHECK_INTERVAL == 0 {
            self.flush_nodes(context);
            if context.stop_signal.load(Relaxed)
                || context.abort_signal.load(Relaxed)
                || self.elapsed_ms() >= self.hard_limit_ms
                || context.nodes_searched.load(Relaxed) >= self.max_nodes
            {
                self.stopped = true;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        board: &mut Board,
        context: &SearchContext,
        mut depth: i32,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        cut_node: bool,
    ) -> i32 {
        self.principal_variation_length[ply] = 0;
        if self.stopped {
            return 0;
        }

        let root = ply == 0;
        let pv_node = beta - alpha > 1;
        if !root {
            if board.get_halfmove_clock() >= 100 || board.is_repetition() {
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return self.evaluate(board, ply);
            }
            // Mate distance pruning: no line from here can beat a mate we already found closer
            // to the root
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let in_check = move_generation::in_check(board);

        // Check extension
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiescence(board, context, alpha, beta, ply);
        }
        self.count_node(context);

        let excluded_move = self.excluded_moves[ply];
        let singular_search = excluded_move != Move::null();

        // Probe the transposition table before generating moves
        let tt_entry = if singular_search {
            None
        } else {
            context.transposition_table.get(board.zobrist_hash)
        };
        let mut tt_move = None;
        let mut tt_score = 0;
        if let Some(entry) = tt_entry {
            if entry.best_move != Move::null() {
                tt_move = Some(entry.best_move);
            }
            tt_score = score_from_tt(entry.score, ply);
            if !pv_node
                && entry.depth as i32 >= depth
                && board.get_halfmove_clock() < 90
                && match entry.node_type {
                    EXACT => true,
                    LOWER_BOUND => tt_score >= beta,
                    UPPER_BOUND => tt_score <= alpha,
                    _ => false,
                }
            {
                return tt_score;
            }
        }
        if root && self.root_best_move.is_some() {
            tt_move = self.root_best_move;
        }

        // Static evaluation, reused from the TT when possible and sharpened by a TT bound
        let static_eval;
        let mut eval;
        if in_check {
            static_eval = -INFINITY;
            eval = -INFINITY;
        } else if singular_search {
            static_eval = self.eval_stack[ply];
            eval = static_eval;
        } else {
            static_eval = match tt_entry {
                Some(entry) => entry.static_eval,
                None => self.evaluate(board, ply),
            };
            eval = static_eval;
            if let Some(entry) = tt_entry
                && tt_score.abs() < MATE_BOUND
                && match entry.node_type {
                    EXACT => true,
                    LOWER_BOUND => tt_score > eval,
                    UPPER_BOUND => tt_score < eval,
                    _ => false,
                }
            {
                eval = tt_score;
            }
        }
        self.eval_stack[ply] = static_eval;
        let improving = !in_check
            && ply >= 2
            && self.eval_stack[ply - 2] != -INFINITY
            && static_eval > self.eval_stack[ply - 2];
        self.killer_moves[ply + 1] = [Move::null(); 2];

        if !pv_node && !in_check && !singular_search {
            // Reverse futility pruning: we're so far above beta that a shallow search won't
            // bring us back down
            if depth <= 8
                && beta.abs() < MATE_BOUND
                && eval - 80 * (depth - improving as i32) >= beta
            {
                return eval;
            }

            // Null move pruning: if passing still beats beta, a real move almost certainly will
            let after_null_move = ply > 0 && self.move_stack[ply - 1].piece == NO_PIECE;
            if !after_null_move
                && depth >= 3
                && eval >= beta
                && static_eval >= beta
                && board.has_non_pawn_material()
            {
                let reduction = 3 + depth / 4 + ((eval - beta) / 200).min(3);
                self.move_stack[ply] = PlyMove::NONE;
                self.accumulators[ply + 1] = self.accumulators[ply];
                let undo = board.make_null_move();
                let score = -self.negamax(
                    board,
                    context,
                    depth - 1 - reduction,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    !cut_node,
                );
                board.unmake_null_move(undo);
                if self.stopped {
                    return 0;
                }
                if score >= beta {
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }

        // Internal iterative reduction: without a TT move this node is probably poorly ordered,
        // so search it shallower rather than spend full depth on it
        if depth >= 4 && tt_move.is_none() && (pv_node || cut_node) && !singular_search {
            depth -= 1;
        }

        let mut moves = move_generation::legal_moves_with_metadata(board);
        if moves.length == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let side = if board.white_to_move() { 0 } else { 1 };
        let counter_move = match self.previous_move(ply, 1) {
            Some(_) => {
                let previous = self.move_stack[ply - 1];
                self.counter_moves[previous.piece][previous.to]
            }
            None => Move::null(),
        };
        let mut scores = [0i32; 218];
        self.score_moves(board, &moves, &mut scores, tt_move, counter_move, ply, side);

        let mut best_score = -INFINITY;
        let mut best_move = Move::null();
        let mut moves_searched = 0;
        let mut quiets_tried = [(Move::null(), 0usize); 64];
        let mut quiet_count = 0usize;
        let mut captures_tried = [(Move::null(), 0usize, 0usize); 32];
        let mut capture_count = 0usize;
        let late_move_limit = (3 + depth * depth) / (2 - improving as i32);

        for i in 0..moves.length {
            let (mov, capture) = pick_next_move(&mut moves, &mut scores, i);
            if mov == excluded_move {
                continue;
            }
            let is_quiet = capture.is_none() && !is_promotion(&mov);
            let piece = piece_key(board, &mov);
            let quiet_history = if is_quiet {
                self.quiet_history(side, piece, &mov, ply)
            } else {
                0
            };

            if !root && best_score > -MATE_BOUND {
                if is_quiet && !in_check {
                    // Late move pruning: quiet moves this late in the list rarely matter
                    if !pv_node && depth <= 8 && quiet_count as i32 >= late_move_limit {
                        continue;
                    }
                    // Futility pruning: too far below alpha for a quiet move to catch up
                    if depth <= 6 && static_eval + 100 + 100 * depth <= alpha {
                        continue;
                    }
                }
                // SEE pruning: skip moves that lose too much material at shallow depth
                if depth <= 8 {
                    let threshold = if is_quiet {
                        -60 * depth
                    } else {
                        -25 * depth * depth
                    };
                    if evaluation::static_exchange_evaluation(board, &mov) < threshold {
                        continue;
                    }
                }
            }

            // Singular extension: if every other move fails well below the TT score, the TT
            // move is the only good one, so search it deeper
            let mut extension = 0;
            if let Some(entry) = tt_entry
                && !root
                && Some(mov) == tt_move
                && depth >= 8
                && entry.depth as i32 >= depth - 3
                && entry.node_type != UPPER_BOUND
                && tt_score.abs() < MATE_BOUND
                && (ply as i32) < 2 * self.root_depth
            {
                let singular_beta = tt_score - 2 * depth;
                self.excluded_moves[ply] = mov;
                let score = self.negamax(
                    board,
                    context,
                    (depth - 1) / 2,
                    ply,
                    singular_beta - 1,
                    singular_beta,
                    cut_node,
                );
                self.excluded_moves[ply] = Move::null();
                self.principal_variation_length[ply] = 0;
                if self.stopped {
                    return 0;
                }
                if score < singular_beta {
                    extension = 1;
                } else if singular_beta >= beta {
                    // Multi-cut: even without the TT move, more than one move beats beta
                    return singular_beta;
                }
            }

            self.move_stack[ply] = PlyMove {
                piece,
                to: mov.ending_index() as usize,
            };
            let metadata = self.make_move(board, &mov, ply);
            let new_depth = depth - 1 + extension;
            let score = if moves_searched == 0 {
                -self.negamax(
                    board,
                    context,
                    new_depth,
                    ply + 1,
                    -beta,
                    -alpha,
                    !pv_node && !cut_node,
                )
            } else {
                // Late move reductions for quiet moves that are ordered late
                let mut reduction = 0;
                if depth >= 3 && is_quiet && moves_searched >= 2 {
                    reduction =
                        self.late_move_reductions[(depth as usize).min(63)][moves_searched.min(63)];
                    if pv_node {
                        reduction -= 1;
                    }
                    if cut_node {
                        reduction += 1;
                    }
                    if !improving {
                        reduction += 1;
                    }
                    if in_check {
                        reduction -= 1;
                    }
                    if self.killer_moves[ply].contains(&mov) || mov == counter_move {
                        reduction -= 1;
                    }
                    reduction -= quiet_history / 8192;
                    reduction = reduction.clamp(0, new_depth - 1);
                }
                // Principal variation search: prove the move is no better with a null window
                let mut score = -self.negamax(
                    board,
                    context,
                    new_depth - reduction,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    true,
                );
                if score > alpha && reduction > 0 {
                    score = -self.negamax(
                        board,
                        context,
                        new_depth,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        !cut_node,
                    );
                }
                if score > alpha && score < beta {
                    score = -self.negamax(board, context, new_depth, ply + 1, -beta, -alpha, false);
                }
                score
            };
            board.unmake_move(&mov, metadata);
            if self.stopped {
                return 0;
            }
            moves_searched += 1;

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mov;
                    if root {
                        self.root_best_move = Some(mov);
                    }
                    self.update_principal_variation(ply, mov);
                    if alpha >= beta {
                        let bonus = (16 * depth * depth + 32 * depth).min(1200);
                        if is_quiet {
                            self.store_killer(ply, mov);
                            if self.previous_move(ply, 1).is_some() {
                                let previous = self.move_stack[ply - 1];
                                self.counter_moves[previous.piece][previous.to] = mov;
                            }
                            self.update_quiet_histories(side, piece, mov, ply, bonus);
                            for &(quiet, quiet_piece) in &quiets_tried[..quiet_count] {
                                self.update_quiet_histories(side, quiet_piece, quiet, ply, -bonus);
                            }
                        } else if let Some(victim) = capture {
                            update_with_gravity(
                                &mut self.capture_history[piece][mov.ending_index() as usize]
                                    [victim as usize],
                                bonus,
                            );
                        }
                        // A cutoff by any move means the captures tried before it were worse
                        for &(tried, tried_piece, victim) in &captures_tried[..capture_count] {
                            update_with_gravity(
                                &mut self.capture_history[tried_piece]
                                    [tried.ending_index() as usize][victim],
                                -bonus,
                            );
                        }
                        break;
                    }
                }
            }
            if is_quiet {
                if quiet_count < quiets_tried.len() {
                    quiets_tried[quiet_count] = (mov, piece);
                    quiet_count += 1;
                }
            } else if let Some(victim) = capture
                && capture_count < captures_tried.len()
            {
                captures_tried[capture_count] = (mov, piece, victim as usize);
                capture_count += 1;
            }
        }

        if moves_searched == 0 {
            // Only possible in a singular search where the excluded move was the only one
            return alpha;
        }

        if !singular_search {
            let node_type = if best_score >= beta {
                LOWER_BOUND
            } else if best_move != Move::null() {
                EXACT
            } else {
                UPPER_BOUND
            };
            context.transposition_table.insert(
                board.zobrist_hash,
                EntryResult::new(
                    depth.clamp(0, u8::MAX as i32) as u8,
                    score_to_tt(best_score, ply),
                    if in_check { 0 } else { static_eval },
                    node_type,
                    best_move,
                ),
            );
        }
        best_score
    }

    fn quiescence(
        &mut self,
        board: &mut Board,
        context: &SearchContext,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        self.principal_variation_length[ply] = 0;
        if self.stopped {
            return 0;
        }
        self.count_node(context);

        if ply >= MAX_PLY - 1 {
            return self.evaluate(board, ply);
        }
        let pv_node = beta - alpha > 1;

        let tt_entry = context.transposition_table.get(board.zobrist_hash);
        let mut tt_move = None;
        if let Some(entry) = tt_entry {
            if entry.best_move != Move::null() {
                tt_move = Some(entry.best_move);
            }
            let tt_score = score_from_tt(entry.score, ply);
            if !pv_node
                && match entry.node_type {
                    EXACT => true,
                    LOWER_BOUND => tt_score >= beta,
                    UPPER_BOUND => tt_score <= alpha,
                    _ => false,
                }
            {
                return tt_score;
            }
        }

        // Captures only, unless in check: then every evasion
        let mut moves = move_generation::legal_captures_with_metadata(board);
        let in_check = moves.check;
        if in_check && moves.length == 0 {
            return -MATE + ply as i32;
        }

        // Standing pat is only allowed when not in check; in check every evasion is searched
        let mut best_score = -INFINITY;
        let mut static_eval = -INFINITY;
        if !in_check {
            static_eval = match tt_entry {
                Some(entry) => entry.static_eval,
                None => self.evaluate(board, ply),
            };
            if static_eval >= beta {
                return static_eval;
            }
            best_score = static_eval;
            alpha = alpha.max(static_eval);
            if moves.length == 0 {
                return static_eval;
            }
        }

        let mut scores = [0i32; 218];
        for i in 0..moves.length {
            let (mov, capture) = moves.moves[i];
            scores[i] = if Some(mov) == tt_move {
                1 << 30
            } else if capture.is_some() || is_promotion(&mov) {
                (1 << 27) + self.capture_order_score(board, &mov, capture)
            } else {
                0
            };
        }

        let mut best_move = Move::null();
        for i in 0..moves.length {
            let (mov, capture) = pick_next_move(&mut moves, &mut scores, i);
            if !in_check {
                // Only queen promotions are worth looking at among quiet promotions
                if capture.is_none() && !is_queen_promotion(&mov) {
                    continue;
                }
                if let Some(victim) = capture {
                    // Delta pruning: even winning this piece for free can't reach alpha
                    if !is_promotion(&mov)
                        && static_eval + evaluation::PIECE_VALUES[victim as usize] + 200 <= alpha
                    {
                        continue;
                    }
                    if evaluation::static_exchange_evaluation(board, &mov) < 0 {
                        continue;
                    }
                }
            }

            let metadata = self.make_move(board, &mov, ply);
            let score = -self.quiescence(board, context, -beta, -alpha, ply + 1);
            board.unmake_move(&mov, metadata);
            if self.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mov;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        let node_type = if best_score >= beta {
            LOWER_BOUND
        } else {
            UPPER_BOUND
        };
        context.transposition_table.insert(
            board.zobrist_hash,
            EntryResult::new(
                0,
                score_to_tt(best_score, ply),
                if in_check { 0 } else { static_eval },
                node_type,
                best_move,
            ),
        );
        best_score
    }

    /// Makes a move and updates the NNUE accumulator for the next ply.
    #[inline(always)]
    fn make_move(&mut self, board: &mut Board, mov: &Move, ply: usize) -> (u8, u16, u64, Option<u8>) {
        let before = PieceBoards::of(board);
        let metadata = board.make_move(mov);
        let after = PieceBoards::of(board);
        let mut removed = [(0, 0, 0); 4];
        let mut added = [(0, 0, 0); 4];
        let (r, a) = before.diff(&after, &mut removed, &mut added);
        let (parents, children) = self.accumulators.split_at_mut(ply + 1);
        children[0].update(&parents[ply], board, &removed[..r], &added[..a]);
        metadata
    }

    #[inline(always)]
    fn evaluate(&self, board: &Board, ply: usize) -> i32 {
        self.accumulators[ply].evaluate(board.white_to_move(), board.all_pieces().count_ones())
    }

    /// The continuation-history key of the move made `back` plies before `ply`.
    fn previous_move(&self, ply: usize, back: usize) -> Option<usize> {
        if ply >= back {
            self.move_stack[ply - back].key()
        } else {
            None
        }
    }

    fn quiet_history(&self, side: usize, piece: usize, mov: &Move, ply: usize) -> i32 {
        let to = mov.ending_index() as usize;
        let current = piece * 64 + to;
        let mut score =
            self.history[side][mov.starting_index() as usize][mov.ending_index() as usize];
        for back in [1, 2] {
            if let Some(previous) = self.previous_move(ply, back) {
                score += self.continuation_history[previous * CONTINUATION_SIZE + current];
            }
        }
        score
    }

    fn update_quiet_histories(
        &mut self,
        side: usize,
        piece: usize,
        mov: Move,
        ply: usize,
        bonus: i32,
    ) {
        update_with_gravity(
            &mut self.history[side][mov.starting_index() as usize][mov.ending_index() as usize],
            bonus,
        );
        let current = piece * 64 + mov.ending_index() as usize;
        for back in [1, 2] {
            if let Some(previous) = self.previous_move(ply, back) {
                update_with_gravity(
                    &mut self.continuation_history[previous * CONTINUATION_SIZE + current],
                    bonus,
                );
            }
        }
    }

    /// MVV-LVA plus capture history, for ordering captures among themselves.
    fn capture_order_score(&self, board: &Board, mov: &Move, capture: Option<u8>) -> i32 {
        let mut score = capture_score(board, mov, capture);
        if let Some(victim) = capture
            && victim != KING
        {
            score += self.capture_history[piece_key(board, mov)][mov.ending_index() as usize]
                [victim as usize]
                / 16;
        }
        score
    }

    /// Scores every move for ordering: TT move, good captures and promotions, killers, the
    /// counter move, quiet moves by history, then losing captures.
    #[allow(clippy::too_many_arguments)]
    fn score_moves(
        &self,
        board: &Board,
        moves: &MovesWithMetadata,
        scores: &mut [i32; 218],
        tt_move: Option<Move>,
        counter_move: Move,
        ply: usize,
        side: usize,
    ) {
        for i in 0..moves.length {
            let (mov, capture) = moves.moves[i];
            scores[i] = if Some(mov) == tt_move {
                1 << 30
            } else if capture.is_some() || is_promotion(&mov) {
                let order = self.capture_order_score(board, &mov, capture);
                if capture.is_some() && evaluation::static_exchange_evaluation(board, &mov) < 0 {
                    // Losing captures go after the quiet moves
                    -(1 << 28) + order
                } else {
                    (1 << 27) + order
                }
            } else if mov == self.killer_moves[ply][0] {
                (1 << 26) + 2
            } else if mov == self.killer_moves[ply][1] {
                (1 << 26) + 1
            } else if mov == counter_move {
                1 << 26
            } else {
                self.quiet_history(side, piece_key(board, &mov), &mov, ply)
            };
        }
    }

    fn store_killer(&mut self, ply: usize, mov: Move) {
        if self.killer_moves[ply][0] != mov {
            self.killer_moves[ply][1] = self.killer_moves[ply][0];
            self.killer_moves[ply][0] = mov;
        }
    }

    fn update_principal_variation(&mut self, ply: usize, mov: Move) {
        self.principal_variation[ply][0] = mov;
        let child_length = if ply + 1 < MAX_PLY {
            self.principal_variation_length[ply + 1]
        } else {
            0
        };
        for i in 0..child_length {
            self.principal_variation[ply][i + 1] = self.principal_variation[ply + 1][i];
        }
        self.principal_variation_length[ply] = child_length + 1;
    }
}

/// History gravity keeps values within ±MAX_HISTORY.
fn update_with_gravity(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
}

/// The moving piece's type and color, as an index for the history tables.
fn piece_key(board: &Board, mov: &Move) -> usize {
    let from = mov.starting_index();
    let piece = board.get_piece_at_square(from).unwrap_or(PAWN) as usize;
    let black = board.black & (1 << from) != 0;
    piece + if black { 6 } else { 0 }
}

/// Selection sort step: swaps the best remaining move into position `index` and returns it.
fn pick_next_move(
    moves: &mut MovesWithMetadata,
    scores: &mut [i32; 218],
    index: usize,
) -> (Move, Option<u8>) {
    let mut best = index;
    for j in index + 1..moves.length {
        if scores[j] > scores[best] {
            best = j;
        }
    }
    scores.swap(index, best);
    moves.moves.swap(index, best);
    moves.moves[index]
}

fn is_promotion(mov: &Move) -> bool {
    mov.flags() & 0b0100 != 0
}

fn is_queen_promotion(mov: &Move) -> bool {
    mov.flags() == 0b0100
}

/// MVV-LVA ordering score for captures and promotions.
fn capture_score(board: &Board, mov: &Move, capture: Option<u8>) -> i32 {
    let attacker = board
        .get_piece_at_square(mov.starting_index())
        .unwrap_or(PAWN) as usize;
    let mut score = match capture {
        Some(victim) if victim != KING => evaluation::mvv_lva(attacker, victim as usize),
        _ => 0,
    };
    if is_queen_promotion(mov) {
        score += evaluation::PIECE_VALUES[4];
    }
    score
}

/// Mate scores are stored relative to the node rather than the root, so they stay correct
/// when the position is reached at a different ply.
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}
