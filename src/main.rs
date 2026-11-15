mod move_generation;
mod search_threads;

use crate::search_threads::{DEFAULT_HASH_MB, DEFAULT_THREADS, Limits, SearchThreads};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, channel};
use std::{process, thread};

fn main() {
    if cfg!(not(debug_assertions)) {
        std::panic::set_hook(Box::new(|panic_info| {
            eprintln!("Critical engine panic: {panic_info}");

            // Immediate termination of the entire engine process
            // We want a fast exit instead of unwinding through the engine.
            process::exit(1);
        }));
    }

    // `HydraChess bench [depth]` runs the benchmark and exits
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(BENCH_DEPTH);
        SearchThreads::bench(depth);
        return;
    }
    // `HydraChess tune <data> [epochs]` fits the evaluation weights and prints eval_params.rs
    if args.get(1).map(String::as_str) == Some("tune") {
        let Some(path) = args.get(2) else {
            eprintln!("usage: HydraChess tune <data file> [epochs]");
            process::exit(2);
        };
        let epochs = args.get(3).and_then(|e| e.parse().ok()).unwrap_or(1000);
        search_threads::tune(path, epochs);
        return;
    }

    // `HydraChess datagen <out> [positions] [threads] [soft nodes]` writes self-play training data
    if args.get(1).map(String::as_str) == Some("datagen") {
        let Some(path) = args.get(2) else {
            eprintln!("usage: HydraChess datagen <out file> [positions] [threads] [soft nodes]");
            process::exit(2);
        };
        let positions = args.get(3).and_then(|p| p.parse().ok()).unwrap_or(1_000_000);
        let threads = args.get(4).and_then(|t| t.parse().ok()).unwrap_or(1);
        let soft_nodes = args.get(5).and_then(|n| n.parse().ok()).unwrap_or(5000);
        search_threads::datagen(path, positions, threads, soft_nodes);
        return;
    }

    // `HydraChess train <out.nnue> <epochs> <data.bin>...` trains the NNUE
    if args.get(1).map(String::as_str) == Some("train") {
        let (Some(out), Some(epochs)) = (args.get(2), args.get(3).and_then(|e| e.parse().ok()))
        else {
            eprintln!("usage: HydraChess train <out.nnue> <epochs> <data.bin>...");
            process::exit(2);
        };
        search_threads::train(out, epochs, &args[4..]);
        return;
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin_lines = stdin.lock().lines();

    let (cmd_tx, cmd_rx) = channel::<EngineCommand>();
    let stop_signal = Arc::new(AtomicBool::new(false));

    let stop_signal_clone = stop_signal.clone();
    let search_thread_handle = thread::spawn(move || search_loop(cmd_rx, stop_signal_clone));

    // UCI main loop: read commands from stdin line by line.
    while let Some(Ok(line)) = stdin_lines.next() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = parts.first() else {
            continue;
        };

        match command {
            "uci" => {
                let mut out = stdout.lock();
                writeln!(out, "id name HydraChess {}", env!("CARGO_PKG_VERSION")).unwrap();
                writeln!(out, "id author Hydra").unwrap();
                writeln!(
                    out,
                    "option name Threads type spin default {DEFAULT_THREADS} min 1 max 256"
                )
                .unwrap();
                writeln!(
                    out,
                    "option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 65536"
                )
                .unwrap();
                writeln!(out, "uciok").unwrap();
                out.flush().unwrap();
            }
            "isready" => {
                let mut out = stdout.lock();
                writeln!(out, "readyok").unwrap();
                out.flush().unwrap();
            }
            "setoption" => {
                // setoption name <name> value <value>
                let name_index = parts.iter().position(|&p| p == "name");
                let value_index = parts.iter().position(|&p| p == "value");
                if let (Some(n), Some(v)) = (name_index, value_index)
                    && n < v
                {
                    let name = parts[n + 1..v].join(" ").to_lowercase();
                    if let Some(value) = parts.get(v + 1).and_then(|v| v.parse().ok()) {
                        cmd_tx.send(EngineCommand::SetOption { name, value }).unwrap();
                    }
                }
            }
            "ucinewgame" => {
                cmd_tx.send(EngineCommand::NewGame).unwrap();
            }
            "position" => {
                let position = parts[1..].join(" ");
                cmd_tx.send(EngineCommand::Position { position }).unwrap();
            }
            "go" => {
                stop_signal.store(false, Relaxed);
                cmd_tx.send(EngineCommand::Go(parse_go(&parts))).unwrap();
            }
            "bench" => {
                let depth = parts.get(1).and_then(|d| d.parse().ok()).unwrap_or(BENCH_DEPTH);
                cmd_tx.send(EngineCommand::Bench { depth }).unwrap();
            }
            "debug" | "d" => {
                cmd_tx.send(EngineCommand::Debug).unwrap();
            }
            "stop" => {
                stop_signal.store(true, Relaxed);
            }
            "quit" => {
                stop_signal.store(true, Relaxed);
                cmd_tx.send(EngineCommand::Quit).unwrap();
                search_thread_handle.join().unwrap();
                break;
            }
            _ => {
                // Ignore unknown commands silently (common in UCI).
            }
        }
    }
}

const BENCH_DEPTH: u8 = 9;

/// Parses `go` parameters; missing or malformed values are ignored instead of panicking.
fn parse_go(parts: &[&str]) -> Limits {
    let value = |key: &str| -> Option<u64> {
        let index = parts.iter().position(|&p| p == key)?;
        parts.get(index + 1)?.parse().ok()
    };
    Limits {
        white_time_ms: value("wtime"),
        black_time_ms: value("btime"),
        white_increment_ms: value("winc").unwrap_or(0),
        black_increment_ms: value("binc").unwrap_or(0),
        movetime_ms: value("movetime"),
        depth: value("depth").map(|d| d.min(u8::MAX as u64) as u8),
        nodes: value("nodes"),
        infinite: parts.contains(&"infinite"),
        ..Limits::default()
    }
}

enum EngineCommand {
    NewGame,
    Position { position: String },
    Go(Limits),
    SetOption { name: String, value: usize },
    Bench { depth: u8 },
    Debug,
    Quit,
}

fn search_loop(cmd_rx: Receiver<EngineCommand>, stop_signal: Arc<AtomicBool>) {
    let mut threads = DEFAULT_THREADS;
    let mut hash_mb = DEFAULT_HASH_MB;
    let mut search = SearchThreads::new(stop_signal.clone(), threads, hash_mb);
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            EngineCommand::NewGame => search.new_game(),
            EngineCommand::Position { position } => search.position(&position),
            EngineCommand::Go(limits) => search.go(limits),
            EngineCommand::SetOption { name, value } => {
                match name.as_str() {
                    "threads" => threads = value,
                    "hash" => hash_mb = value,
                    _ => continue,
                }
                // Drop the old threads first so their memory is freed before reallocating
                drop(search);
                search = SearchThreads::new(stop_signal.clone(), threads, hash_mb);
            }
            EngineCommand::Bench { depth } => SearchThreads::bench(depth),
            EngineCommand::Debug => search.debug(),
            EngineCommand::Quit => break,
        }
    }
}
