//! Texel tuning: fits the evaluation weights to game results.
//!
//! Input is one position per line, `FEN | result`, where the result is from white's point of
//! view (1, 0.5 or 0), as written by `scripts/match.py datagen`. The evaluation is linear in
//! its weights, so each position is reduced once to its feature counts, and the weights are
//! then fitted with Adam on the mean squared error between the game result and
//! `sigmoid(K * eval)`. The tuned weights are printed to stdout as a new `eval_params.rs`.

use super::eval_params::PARAMS;
use super::evaluation::{
    self, BISHOP_MOBILITY, BISHOP_PAIR, DOUBLED_PAWN, ISOLATED_PAWN, KING_ATTACK, KING_SHIELD,
    KNIGHT_MOBILITY, MATERIAL, MAX_PHASE, NUM_PARAMS, PASSED_PAWN, PSQT, QUEEN_MOBILITY,
    ROOK_MOBILITY, ROOK_OPEN_FILE, ROOK_SEMI_OPEN_FILE, TEMPO, Trace,
};
use crate::move_generation::{self, Board};
use std::fs;
use std::thread;

struct DenseTrace(Vec<i32>);

impl Trace for DenseTrace {
    fn add(&mut self, index: usize, count: i32) {
        self.0[index] += count;
    }
}

struct Position {
    /// Non-zero (parameter, count) pairs
    features: Vec<(u16, i16)>,
    /// Middlegame weight, phase / MAX_PHASE
    middlegame: f64,
    result: f64,
}

type Weights = Vec<[f64; 2]>;

fn evaluate(position: &Position, weights: &Weights) -> f64 {
    let mut mg = 0.0;
    let mut eg = 0.0;
    for &(index, count) in &position.features {
        mg += weights[index as usize][0] * count as f64;
        eg += weights[index as usize][1] * count as f64;
    }
    mg * position.middlegame + eg * (1.0 - position.middlegame)
}

fn sigmoid(k: f64, eval: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-k * eval / 400.0))
}

/// Positions where a capture wins material aren't quiet, so their static eval is misleading.
fn is_quiet(board: &Board) -> bool {
    if move_generation::in_check(board) {
        return false;
    }
    let captures = move_generation::legal_captures_with_metadata(board);
    (0..captures.length).all(|i| {
        let (mov, capture) = captures.moves[i];
        capture.is_none() || evaluation::static_exchange_evaluation(board, &mov) <= 0
    })
}

fn load(path: &str) -> Vec<Position> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("can't read {path}: {e}"));
    let mut positions = Vec::new();
    let mut skipped = 0;
    for line in text.lines() {
        let Some((fen, result)) = line.split_once('|') else {
            continue;
        };
        let Ok(result) = result.trim().parse::<f64>() else {
            continue;
        };
        let board = Board::from_fen(fen.trim());
        if !is_quiet(&board) {
            skipped += 1;
            continue;
        }
        let mut trace = DenseTrace(vec![0; NUM_PARAMS]);
        evaluation::evaluate_traced(&board, &mut trace);
        let features: Vec<(u16, i16)> = trace
            .0
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count != 0)
            .map(|(index, &count)| (index as u16, count as i16))
            .collect();
        positions.push(Position {
            features,
            middlegame: evaluation::game_phase(&board) as f64 / MAX_PHASE as f64,
            result,
        });
    }
    eprintln!(
        "loaded {} quiet positions ({skipped} skipped)",
        positions.len()
    );
    positions
}

fn mean_error(positions: &[Position], weights: &Weights, k: f64) -> f64 {
    let total: f64 = parallel_chunks(positions, |chunk| {
        chunk
            .iter()
            .map(|p| (p.result - sigmoid(k, evaluate(p, weights))).powi(2))
            .sum::<f64>()
    })
    .into_iter()
    .sum();
    total / positions.len() as f64
}

fn parallel_chunks<T: Send>(
    positions: &[Position],
    work: impl Fn(&[Position]) -> T + Sync,
) -> Vec<T> {
    let threads = thread::available_parallelism().map_or(4, |n| n.get());
    let chunk_size = positions.len().div_ceil(threads).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = positions
            .chunks(chunk_size)
            .map(|chunk| scope.spawn(|| work(chunk)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Finds the K that best maps the current evaluation onto results.
fn fit_k(positions: &[Position], weights: &Weights) -> f64 {
    let (mut low, mut high) = (0.1, 4.0);
    for _ in 0..40 {
        let a = low + (high - low) / 3.0;
        let b = high - (high - low) / 3.0;
        if mean_error(positions, weights, a) < mean_error(positions, weights, b) {
            high = b;
        } else {
            low = a;
        }
    }
    (low + high) / 2.0
}

fn gradient(positions: &[Position], weights: &Weights, k: f64) -> Weights {
    let partials = parallel_chunks(positions, |chunk| {
        let mut gradient = vec![[0.0; 2]; NUM_PARAMS];
        for position in chunk {
            let s = sigmoid(k, evaluate(position, weights));
            // d(error)/d(eval), up to a constant factor that Adam doesn't care about
            let d = (s - position.result) * s * (1.0 - s);
            for &(index, count) in &position.features {
                let g = d * count as f64;
                gradient[index as usize][0] += g * position.middlegame;
                gradient[index as usize][1] += g * (1.0 - position.middlegame);
            }
        }
        gradient
    });
    let mut total = vec![[0.0; 2]; NUM_PARAMS];
    for partial in partials {
        for (t, p) in total.iter_mut().zip(partial) {
            t[0] += p[0];
            t[1] += p[1];
        }
    }
    total
}

pub fn tune(path: &str, epochs: usize) {
    let positions = load(path);
    if positions.is_empty() {
        eprintln!("no positions to tune on");
        return;
    }
    let mut weights: Weights = PARAMS.iter().map(|s| [s.0 as f64, s.1 as f64]).collect();
    let k = fit_k(&positions, &weights);
    eprintln!(
        "K = {k:.4}, starting error {:.6}",
        mean_error(&positions, &weights, k)
    );

    const LEARNING_RATE: f64 = 1.0;
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    let mut momentum = vec![[0.0; 2]; NUM_PARAMS];
    let mut velocity = vec![[0.0; 2]; NUM_PARAMS];
    for epoch in 1..=epochs {
        let gradient = gradient(&positions, &weights, k);
        for i in 0..NUM_PARAMS {
            for phase in 0..2 {
                let g = gradient[i][phase];
                momentum[i][phase] = BETA1 * momentum[i][phase] + (1.0 - BETA1) * g;
                velocity[i][phase] = BETA2 * velocity[i][phase] + (1.0 - BETA2) * g * g;
                let m = momentum[i][phase] / (1.0 - BETA1.powi(epoch as i32));
                let v = velocity[i][phase] / (1.0 - BETA2.powi(epoch as i32));
                weights[i][phase] -= LEARNING_RATE * m / (v.sqrt() + 1e-8);
            }
        }
        if epoch % 50 == 0 || epoch == epochs {
            eprintln!(
                "epoch {epoch}: error {:.6}",
                mean_error(&positions, &weights, k)
            );
        }
    }
    print!("{}", format_params(&weights));
}

fn format_params(weights: &Weights) -> String {
    let s = |i: usize| {
        format!(
            "S({}, {})",
            weights[i][0].round() as i32,
            weights[i][1].round() as i32
        )
    };
    let row = |out: &mut String, range: std::ops::Range<usize>| {
        let items: Vec<String> = range.map(s).collect();
        out.push_str(&format!("    {},\n", items.join(", ")));
    };
    let mut out = String::from(
        "//! Evaluation weights, as (middlegame, endgame) pairs.\n//!\n\
         //! `HydraChess tune <data>` prints a replacement for this whole file.\n\n\
         use super::evaluation::S;\n\n#[rustfmt::skip]\n",
    );
    out.push_str(&format!("pub const PARAMS: [S; {NUM_PARAMS}] = [\n"));
    out.push_str("    // Material\n");
    row(&mut out, MATERIAL..MATERIAL + 6);
    for (piece, name) in ["pawn", "knight", "bishop", "rook", "queen", "king"]
        .iter()
        .enumerate()
    {
        out.push_str(&format!(
            "    // Piece-square table: {name} (a1..h8, white's point of view)\n"
        ));
        for rank in 0..8 {
            let start = PSQT + piece * 64 + rank * 8;
            row(&mut out, start..start + 8);
        }
    }
    out.push_str("    // Passed pawn by relative rank\n");
    row(&mut out, PASSED_PAWN..PASSED_PAWN + 8);
    for (name, start, length) in [
        ("knight", KNIGHT_MOBILITY, 9),
        ("bishop", BISHOP_MOBILITY, 14),
        ("rook", ROOK_MOBILITY, 15),
        ("queen", QUEEN_MOBILITY, 28),
    ] {
        out.push_str(&format!("    // Mobility: {name}\n"));
        let mut i = start;
        while i < start + length {
            let end = (i + 8).min(start + length);
            row(&mut out, i..end);
            i = end;
        }
    }
    for (name, index) in [
        ("Bishop pair", BISHOP_PAIR),
        ("Doubled pawn", DOUBLED_PAWN),
        ("Isolated pawn", ISOLATED_PAWN),
        ("Rook on open file", ROOK_OPEN_FILE),
        ("Rook on semi-open file", ROOK_SEMI_OPEN_FILE),
        ("King pawn shield", KING_SHIELD),
    ] {
        out.push_str(&format!("    // {name}\n"));
        row(&mut out, index..index + 1);
    }
    out.push_str("    // King zone attacks: knight, bishop, rook, queen\n");
    row(&mut out, KING_ATTACK..KING_ATTACK + 4);
    out.push_str("    // Tempo\n");
    row(&mut out, TEMPO..TEMPO + 1);
    out.push_str("];\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traced_features_reproduce_the_evaluation() {
        let weights: Weights = PARAMS.iter().map(|s| [s.0 as f64, s.1 as f64]).collect();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/pp3k2/2p2p2/5p2/2PP4/1P3P2/P4K2/8 w - - 0 35",
            "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R b KQ - 0 9",
        ] {
            let board = Board::from_fen(fen);
            let mut trace = DenseTrace(vec![0; NUM_PARAMS]);
            let eval = evaluation::evaluate_traced(&board, &mut trace);
            let position = Position {
                features: trace
                    .0
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c != 0)
                    .map(|(i, &c)| (i as u16, c as i16))
                    .collect(),
                middlegame: evaluation::game_phase(&board) as f64 / MAX_PHASE as f64,
                result: 0.5,
            };
            assert!(
                (evaluate(&position, &weights) - eval as f64).abs() <= 1.0,
                "{fen}"
            );
        }
    }

    #[test]
    fn printed_params_round_trip_the_current_file() {
        let weights: Weights = PARAMS.iter().map(|s| [s.0 as f64, s.1 as f64]).collect();
        let current = include_str!("eval_params.rs");
        assert_eq!(format_params(&weights), current);
    }
}
