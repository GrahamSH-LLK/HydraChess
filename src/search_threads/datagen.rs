//! Self-play data generation for training the NNUE.
//!
//! `HydraChess datagen <out file> [positions] [threads] [soft nodes]` plays games from random
//! openings at a fixed node budget and appends one packed 32-byte record per quiet position
//! (see `PackedPosition`). Scores and results are from white's point of view.

use super::searcher::{Limits, MATE, Search, SearchContext};
use super::transposition_table::TranspositionTable;
use crate::move_generation::{self, Board};
use minstant::Instant;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::thread;

/// One training position: occupancy, then a nibble per occupied square (ascending square
/// order) holding `piece type | 8 if black`, the search score and game result (white's view),
/// and the side to move.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct PackedPosition {
    pub occupancy: u64,
    pub pieces: [u8; 16],
    pub score: i16,
    /// 0 black win, 1 draw, 2 white win
    pub result: u8,
    /// 0 white, 1 black
    pub side_to_move: u8,
    pub _padding: [u8; 4],
}

const _: () = assert!(size_of::<PackedPosition>() == 32);

impl PackedPosition {
    pub fn pack(board: &Board, score: i16) -> Self {
        let occupancy = board.all_pieces();
        let mut pieces = [0u8; 16];
        let mut bits = occupancy;
        let mut i = 0;
        while bits != 0 {
            let square = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let piece = board.get_piece_at_square(square).unwrap();
            let black = board.black & (1 << square) != 0;
            let nibble = piece | if black { 8 } else { 0 };
            pieces[i / 2] |= nibble << (4 * (i % 2));
            i += 1;
        }
        Self {
            occupancy,
            pieces,
            score,
            result: 1,
            side_to_move: if board.white_to_move() { 0 } else { 1 },
            _padding: [0; 4],
        }
    }

    /// Calls `f(color (0 white), piece type, square)` for every piece.
    #[inline(always)]
    pub fn for_each_piece(&self, mut f: impl FnMut(usize, usize, usize)) {
        let mut bits = self.occupancy;
        let mut i = 0;
        while bits != 0 {
            let square = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let nibble = (self.pieces[i / 2] >> (4 * (i % 2))) & 0xF;
            f((nibble >> 3) as usize, (nibble & 7) as usize, square);
            i += 1;
        }
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}


/// With `deep_multiplier` > 1, one side (random per game) searches that many times more nodes,
/// so the shallow side's misjudgements get punished and show up in the game results. Only the
/// deep side's scores adjudicate. `keep_percent` keeps a random share of the quiet positions, so
/// a fixed position count spans more games.
pub fn datagen(
    path: &str,
    target_positions: u64,
    threads: usize,
    soft_nodes: u64,
    deep_multiplier: u64,
    keep_percent: u64,
) {
    move_generation::move_lookup::initialize();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("can't open {path}: {e}"));
    let writer = Mutex::new(BufWriter::new(file));
    let positions = AtomicU64::new(0);
    let games = AtomicU64::new(0);
    let timer = Instant::now();
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    thread::scope(|scope| {
        for id in 0..threads {
            let writer = &writer;
            let positions = &positions;
            let games = &games;
            scope.spawn(move || {
                let mut rng = Rng(seed ^ (id as u64).wrapping_mul(0x1234_5678_9ABC_DEF1));
                let transposition_table = TranspositionTable::new(16);
                let stop_signal = AtomicBool::new(false);
                let abort_signal = AtomicBool::new(false);
                let nodes_searched = AtomicU64::new(0);
                let mut search = Search::new(1);
                search.print_info = false;
                while positions.load(Relaxed) < target_positions {
                    let records = play_game(
                        &mut rng,
                        &mut search,
                        &transposition_table,
                        &stop_signal,
                        &abort_signal,
                        &nodes_searched,
                        soft_nodes,
                        deep_multiplier,
                        keep_percent,
                    );
                    let mut out = writer.lock().unwrap();
                    for record in &records {
                        out.write_all(record.as_bytes()).unwrap();
                    }
                    drop(out);
                    let total = positions.fetch_add(records.len() as u64, Relaxed)
                        + records.len() as u64;
                    let game_count = games.fetch_add(1, Relaxed) + 1;
                    if game_count % 500 == 0 {
                        let seconds = timer.elapsed().as_secs_f64();
                        eprintln!(
                            "games {game_count} positions {total} ({:.0}/s)",
                            total as f64 / seconds
                        );
                    }
                }
            });
        }
    });
    writer.lock().unwrap().flush().unwrap();
    eprintln!(
        "done: {} games, {} positions",
        games.load(Relaxed),
        positions.load(Relaxed)
    );
}

fn play_game(
    rng: &mut Rng,
    search: &mut Search,
    transposition_table: &TranspositionTable,
    stop_signal: &AtomicBool,
    abort_signal: &AtomicBool,
    nodes_searched: &AtomicU64,
    soft_nodes: u64,
    deep_multiplier: u64,
    keep_percent: u64,
) -> Vec<PackedPosition> {
    transposition_table.clear();
    search.new_game();
    let context = SearchContext {
        transposition_table,
        stop_signal,
        abort_signal,
        nodes_searched,
    };
    let limits = Limits {
        nodes: Some(soft_nodes * 20),
        soft_nodes: Some(soft_nodes),
        ..Limits::default()
    };
    let deep = deep_multiplier > 1;
    let deep_limits = Limits {
        nodes: Some(soft_nodes * deep_multiplier * 20),
        soft_nodes: Some(soft_nodes * deep_multiplier),
        ..Limits::default()
    };
    let deep_is_white = rng.below(2) == 0;
    // Deep games only adjudicate on every other ply, so the counts halve for the same plies
    let (win_needed, draw_needed) = if deep { (4, 6) } else { (8, 12) };

    // Random opening: 8 to 12 random plies, then reject lopsided starts
    let mut board = 'opening: loop {
        let mut board = Board::starting();
        let plies = 8 + rng.below(5);
        for _ in 0..plies {
            let moves = move_generation::legal_moves(&board);
            if moves.length == 0 {
                continue 'opening;
            }
            board.make_move(&moves.moves[rng.below(moves.length)]);
        }
        if move_generation::legal_moves(&board).length == 0 {
            continue;
        }
        nodes_searched.store(0, Relaxed);
        let check = search.search(&mut board.clone(), &context, &Limits {
            depth: Some(8),
            nodes: Some(50_000),
            ..Limits::default()
        });
        if check.score.abs() <= 300 {
            break board;
        }
    };
    transposition_table.clear();

    let mut records = Vec::new();
    let mut win_plies = 0;
    let mut draw_plies = 0;
    let mut ply = 0;
    let result = loop {
        let moves = move_generation::legal_moves_with_metadata(&board);
        if moves.length == 0 {
            break if move_generation::in_check(&board) {
                if board.white_to_move() { 0 } else { 2 }
            } else {
                1
            };
        }
        if board.get_halfmove_clock() >= 100 || board.is_repetition() || board.is_insufficient_material()
        {
            break 1;
        }
        if ply >= 400 {
            break 1;
        }

        transposition_table.new_search();
        nodes_searched.store(0, Relaxed);
        let deep_turn = deep && board.white_to_move() == deep_is_white;
        let found = search.search(
            &mut board,
            &context,
            if deep_turn { &deep_limits } else { &limits },
        );
        let white_score = if board.white_to_move() {
            found.score
        } else {
            -found.score
        };

        // Keep quiet positions with a real score
        let mov = found.best_move;
        let (_, capture) = (0..moves.length)
            .map(|i| moves.moves[i])
            .find(|&(m, _)| m == mov)
            .unwrap_or((mov, None));
        let promotion = mov.flags() & 0b0100 != 0;
        if found.depth > 0
            && capture.is_none()
            && !promotion
            && !moves.check
            && found.score.abs() < MATE - 1000
            && (keep_percent >= 100 || (rng.below(100) as u64) < keep_percent)
        {
            records.push(PackedPosition::pack(&board, white_score.clamp(-32000, 32000) as i16));
        }

        // Adjudication, trusting only the deep side's scores in deep games
        if !deep || deep_turn {
            if white_score.abs() >= 2500 {
                win_plies += 1;
                if win_plies >= win_needed {
                    break if white_score > 0 { 2 } else { 0 };
                }
            } else {
                win_plies = 0;
            }
            if ply >= 80 && white_score.abs() <= 8 {
                draw_plies += 1;
                if draw_plies >= draw_needed {
                    break 1;
                }
            } else {
                draw_plies = 0;
            }
        }

        board.make_move(&mov);
        ply += 1;
    };
    if deep && std::env::var_os("DATAGEN_DEBUG").is_some() {
        eprintln!("game deep={} result={result} plies={ply}", if deep_is_white { "white" } else { "black" });
    }
    for record in &mut records {
        record.result = result;
    }
    records
}
