mod datagen;
mod eval_params;
mod evaluation;
mod nnue;
mod searcher;
mod train;
mod transposition_table;
mod tune;

pub use datagen::datagen;
pub use train::train;
pub use tune::tune;

use super::move_generation;
use super::move_generation::Board;
use crate::move_generation::mov::Move;
pub use crate::search_threads::searcher::Limits;
use crate::search_threads::searcher::{Search, SearchContext};
use crate::search_threads::transposition_table::TranspositionTable;
use minstant::Instant;
use std::io::Write;
use std::io::stdout;
use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Barrier};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

pub struct SearchResult {
    pub depth: u8,
    pub score: i32,
    pub best_move: Move,
}

struct SharedState {
    limits: Mutex<Limits>,
    start_barrier: Barrier,
    end_barrier: Barrier,
    root_board: Mutex<Board>,
    // Transposition table is lock-free to allow TT improvements in one thread to improve searches in another
    transposition_table: TranspositionTable,
    /// The main thread's result, which decides the move
    result: Mutex<SearchResult>,
    nodes_searched: AtomicU64,
    /// Set by the GUI's `stop`
    stop_signal: Arc<AtomicBool>,
    /// Set by the main thread when it finishes, to stop the helper threads
    abort_signal: AtomicBool,
    /// Set when the worker threads should exit
    shutdown: AtomicBool,
    new_game: AtomicBool,
}

pub struct SearchThreads {
    shared_state: Arc<SharedState>,
    workers: Vec<JoinHandle<()>>,
}

pub const DEFAULT_THREADS: usize = 4;
pub const DEFAULT_HASH_MB: usize = 64;

impl SearchThreads {
    pub fn new(stop_signal: Arc<AtomicBool>, threads: usize, hash_mb: usize) -> Self {
        // Warm up the LazyLock'd lookups
        move_generation::move_lookup::initialize();

        let threads = threads.max(1);
        // Coordinator thread is +1
        let total_threads = threads + 1;

        let shared_state = Arc::new(SharedState {
            limits: Mutex::new(Limits::default()),
            start_barrier: Barrier::new(total_threads),
            end_barrier: Barrier::new(total_threads),
            root_board: Mutex::new(Board::starting()),
            transposition_table: TranspositionTable::new(hash_mb.max(1)),
            result: Mutex::new(SearchResult {
                depth: 0,
                score: 0,
                best_move: Move::null(),
            }),
            nodes_searched: AtomicU64::new(0),
            stop_signal,
            abort_signal: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            new_game: AtomicBool::new(false),
        });

        let workers = (0..threads)
            .map(|id| {
                let state = Arc::clone(&shared_state);
                thread::spawn(move || {
                    let mut search = Search::new(id);
                    loop {
                        state.start_barrier.wait();
                        if state.shutdown.load(Relaxed) {
                            break;
                        }
                        if state.new_game.load(Relaxed) {
                            search.new_game();
                        }
                        let mut board = state.root_board.lock().unwrap().clone();
                        let limits = *state.limits.lock().unwrap();
                        let context = SearchContext {
                            transposition_table: &state.transposition_table,
                            stop_signal: &state.stop_signal,
                            abort_signal: &state.abort_signal,
                            nodes_searched: &state.nodes_searched,
                        };

                        let result = search.search(&mut board, &context, &limits);
                        if id == 0 {
                            *state.result.lock().unwrap() = result;
                            // The main thread is done, so the helpers should stop too
                            state.abort_signal.store(true, Relaxed);
                        }
                        // Always reach the end barrier, or the coordinator waits forever
                        state.end_barrier.wait();
                    }
                })
            })
            .collect();

        Self {
            shared_state,
            workers,
        }
    }

    pub fn new_game(&mut self) {
        *self.shared_state.root_board.lock().unwrap() = Board::starting();
        self.shared_state.transposition_table.clear();
        self.shared_state.new_game.store(true, Relaxed);
    }

    pub fn position(&mut self, string: &str) {
        let parts: Vec<&str> = string.split_whitespace().collect();
        let moves_index = parts.iter().position(|&p| p == "moves");
        let position_end = moves_index.unwrap_or(parts.len());

        let board = match parts.first() {
            Some(&"startpos") => Board::starting(),
            Some(&"fen") if position_end > 1 => {
                let mut fen = parts[1..position_end].to_vec();
                // Fill in missing halfmove and fullmove counters (e.g. EPD-style FENs)
                while fen.len() < 6 {
                    fen.push(if fen.len() == 4 { "0" } else { "1" });
                }
                Board::from_fen(&fen.join(" "))
            }
            _ => return,
        };
        let mut board_guard = self.shared_state.root_board.lock().unwrap();
        *board_guard = board;
        if let Some(index) = moves_index {
            for mov in &parts[index + 1..] {
                board_guard.move_from_uci_lan(mov);
            }
        }
    }

    pub fn debug(&self) {
        println!("Board: {}", self.shared_state.root_board.lock().unwrap());
    }

    pub fn go(&mut self, mut limits: Limits) {
        let state = &self.shared_state;
        if state.root_board.lock().unwrap().white_to_move() {
            limits.time_left_ms = limits.white_time_ms;
            limits.increment_ms = limits.white_increment_ms;
        } else {
            limits.time_left_ms = limits.black_time_ms;
            limits.increment_ms = limits.black_increment_ms;
        }
        *state.limits.lock().unwrap() = limits;
        state.nodes_searched.store(0, Relaxed);
        state.abort_signal.store(false, Relaxed);
        state.transposition_table.new_search();
        state.start_barrier.wait();
        state.end_barrier.wait();
        state.new_game.store(false, Relaxed);

        // In infinite mode the best move may only be sent after the GUI says stop
        if limits.infinite {
            while !state.stop_signal.load(Relaxed) {
                thread::sleep(Duration::from_millis(1));
            }
        }

        let best_move = state.result.lock().unwrap().best_move;
        let mut out = stdout().lock();
        if best_move == Move::null() {
            // No legal moves (checkmate or stalemate)
            writeln!(out, "bestmove 0000").unwrap();
        } else {
            writeln!(out, "bestmove {}", best_move).unwrap();
        }
        out.flush().unwrap();
    }

    /// Searches a fixed set of positions to a fixed depth on one thread and reports the total
    /// node count, which is deterministic and useful as a search signature.
    pub fn bench(depth: u8) {
        const POSITIONS: [&str; 12] = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1pn1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
            "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9",
            "8/5pk1/6p1/1R5p/7P/6P1/r4PK1/8 w - - 0 40",
            "8/pp3k2/2p2p2/5p2/2PP4/1P3P2/P4K2/8 w - - 0 35",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        ];
        let transposition_table = TranspositionTable::new(16);
        let stop_signal = AtomicBool::new(false);
        let abort_signal = AtomicBool::new(false);
        let nodes_searched = AtomicU64::new(0);
        let mut total_nodes = 0;
        let timer = Instant::now();
        for fen in POSITIONS {
            transposition_table.clear();
            nodes_searched.store(0, Relaxed);
            let context = SearchContext {
                transposition_table: &transposition_table,
                stop_signal: &stop_signal,
                abort_signal: &abort_signal,
                nodes_searched: &nodes_searched,
            };
            let mut search = Search::new(0);
            search.print_info = false;
            let limits = Limits {
                depth: Some(depth),
                ..Limits::default()
            };
            let result = search.search(&mut Board::from_fen(fen), &context, &limits);
            let nodes = nodes_searched.load(Relaxed);
            println!("{fen}: bestmove {} nodes {nodes}", result.best_move);
            total_nodes += nodes;
        }
        let elapsed_ms = (timer.elapsed().as_millis() as u64).max(1);
        println!(
            "{total_nodes} nodes {} nps {elapsed_ms} ms",
            total_nodes * 1000 / elapsed_ms
        );
    }
}

impl Drop for SearchThreads {
    fn drop(&mut self) {
        // Release the workers from the start barrier with the shutdown flag set, then join them
        self.shared_state.shutdown.store(true, Relaxed);
        self.shared_state.start_barrier.wait();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
