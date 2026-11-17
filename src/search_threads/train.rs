//! NNUE trainer.
//!
//! `HydraChess train <out.nnue> <epochs> <data.bin>...` fits the network in `nnue.rs` to
//! positions from `HydraChess datagen` on the CPU, then writes the quantized network.
//! The target blends the search score and the game result:
//! `(1 - WDL) * sigmoid(score / SCALE) + WDL * result`.

use super::datagen::PackedPosition;
use super::nnue::{
    HIDDEN, INPUTS, Network, OUTPUT_BUCKETS, QA, QB, SCALE, View, output_bucket,
};
use minstant::Instant;
use std::fs;
use std::thread;

const BATCH: usize = 16_384;
const WDL: f32 = 0.3;
const MAX_LR: f32 = 0.001;
const MIN_LR: f32 = 0.00002;
/// Weights are clipped to this so the quantized values fit
const CLIP: f32 = 1.98;

const W1: usize = INPUTS * HIDDEN;
const B1: usize = W1;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + OUTPUT_BUCKETS * 2 * HIDDEN;
const PARAMS: usize = B2 + OUTPUT_BUCKETS;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn load(paths: &[String]) -> Vec<PackedPosition> {
    // Reserve exactly, so a few hundred million positions don't double the memory on growth
    let total: u64 = paths
        .iter()
        .map(|path| fs::metadata(path).map_or(0, |m| m.len() / 32))
        .sum();
    let mut positions = Vec::with_capacity(total as usize);
    for path in paths {
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("can't read {path}: {e}"));
        for chunk in bytes.chunks_exact(32) {
            let mut position = PackedPosition::default();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    &mut position as *mut PackedPosition as *mut u8,
                    32,
                );
            }
            positions.push(position);
        }
    }
    positions
}

/// The side to move's view, then the other side's.
fn position_views(position: &PackedPosition, stm: usize) -> [View; 2] {
    let mut kings = [0; 2];
    position.for_each_piece(|color, piece, square| {
        if piece == 5 {
            kings[color] = square;
        }
    });
    [View::new(stm, kings[stm]), View::new(1 - stm, kings[1 - stm])]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Accumulates gradients over `batch` into `grad`; returns the summed loss.
fn gradients(params: &[f32], batch: &[PackedPosition], grad: &mut [f32]) -> f64 {
    let mut loss = 0.0f64;
    let mut acc = [[0f32; HIDDEN]; 2];
    let mut features = [[0usize; 32]; 2];
    for position in batch {
        let stm = position.side_to_move as usize;
        let views = position_views(position, stm);
        let mut count = 0;
        position.for_each_piece(|color, piece, square| {
            // Perspective 0 is the side to move
            features[0][count] = views[0].feature(stm, color, piece, square);
            features[1][count] = views[1].feature(1 - stm, color, piece, square);
            count += 1;
        });
        let bucket = output_bucket(count as u32);

        let mut out = params[B2 + bucket];
        let w2 = &params[W2 + bucket * 2 * HIDDEN..W2 + (bucket + 1) * 2 * HIDDEN];
        for side in 0..2 {
            acc[side].copy_from_slice(&params[B1..B1 + HIDDEN]);
            for &f in &features[side][..count] {
                let row = &params[f * HIDDEN..(f + 1) * HIDDEN];
                for (a, w) in acc[side].iter_mut().zip(row) {
                    *a += *w;
                }
            }
            let half = &w2[side * HIDDEN..(side + 1) * HIDDEN];
            let mut sum = 0.0;
            for (a, w) in acc[side].iter().zip(half) {
                let c = a.clamp(0.0, 1.0);
                sum += c * c * w;
            }
            out += sum;
        }

        let (score, result) = if stm == 0 {
            (position.score as f32, position.result as f32 / 2.0)
        } else {
            (-(position.score as f32), 1.0 - position.result as f32 / 2.0)
        };
        let target = (1.0 - WDL) * sigmoid(score / SCALE as f32) + WDL * result;
        let predicted = sigmoid(out);
        let error = predicted - target;
        loss += (error * error) as f64;
        let g = 2.0 * error * predicted * (1.0 - predicted);

        grad[B2 + bucket] += g;
        for side in 0..2 {
            let w2_offset = W2 + bucket * 2 * HIDDEN + side * HIDDEN;
            let mut dacc = [0f32; HIDDEN];
            for j in 0..HIDDEN {
                let a = acc[side][j];
                let c = a.clamp(0.0, 1.0);
                grad[w2_offset + j] += g * c * c;
                dacc[j] = if a > 0.0 && a < 1.0 {
                    g * params[w2_offset + j] * 2.0 * a
                } else {
                    0.0
                };
            }
            for (b, d) in grad[B1..B1 + HIDDEN].iter_mut().zip(&dacc) {
                *b += *d;
            }
            for &f in &features[side][..count] {
                let row = &mut grad[f * HIDDEN..(f + 1) * HIDDEN];
                for (r, d) in row.iter_mut().zip(&dacc) {
                    *r += *d;
                }
            }
        }
    }
    loss
}

fn parallel_gradients(
    params: &[f32],
    batch: &[PackedPosition],
    grads: &mut [Vec<f32>],
) -> f64 {
    let chunk = batch.len().div_ceil(grads.len()).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = batch
            .chunks(chunk)
            .zip(grads.iter_mut())
            .map(|(positions, grad)| {
                scope.spawn(move || {
                    grad.iter_mut().for_each(|g| *g = 0.0);
                    gradients(params, positions, grad)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn validation_loss(params: &[f32], positions: &[PackedPosition], grads: &mut [Vec<f32>]) -> f64 {
    let mut loss = 0.0;
    for batch in positions.chunks(BATCH) {
        loss += parallel_gradients(params, batch, grads);
    }
    loss / positions.len().max(1) as f64
}

pub fn train(out_path: &str, epochs: usize, data_paths: &[String]) {
    let threads = thread::available_parallelism().map_or(4, |n| n.get());
    let mut positions = load(data_paths);
    let mut rng = Rng(0xC0FFEE);
    // Shuffle once, then hold out a validation slice
    for i in (1..positions.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        positions.swap(i, j);
    }
    let validation_size = (positions.len() / 100).min(200_000);
    let (validation, training) = positions.split_at_mut(validation_size);
    eprintln!(
        "{} training positions, {} validation, {threads} threads",
        training.len(),
        validation.len()
    );

    let mut params = vec![0f32; PARAMS];
    // He-style init for the feature layer, small output weights
    let w1_scale = (1.0 / 32.0f32).sqrt();
    for p in &mut params[..W1] {
        *p = (rng.uniform() * 2.0 - 1.0) * w1_scale;
    }
    let w2_scale = (1.0 / (2.0 * HIDDEN as f32)).sqrt();
    for p in &mut params[W2..B2] {
        *p = (rng.uniform() * 2.0 - 1.0) * w2_scale;
    }

    let mut momentum = vec![0f32; PARAMS];
    let mut velocity = vec![0f32; PARAMS];
    let mut grads = vec![vec![0f32; PARAMS]; threads];
    let mut total = vec![0f32; PARAMS];
    let (beta1, beta2, epsilon) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0i32;
    let batches_per_epoch = training.len() / BATCH;
    let total_steps = (epochs * batches_per_epoch).max(1);
    let timer = Instant::now();

    for epoch in 0..epochs {
        for i in (1..training.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            training.swap(i, j);
        }
        let mut epoch_loss = 0.0;
        for batch in training.chunks_exact(BATCH) {
            let progress = step as f32 / total_steps as f32;
            let lr = MIN_LR
                + 0.5 * (MAX_LR - MIN_LR) * (1.0 + (std::f32::consts::PI * progress).cos());
            epoch_loss += parallel_gradients(&params, batch, &mut grads);

            // Sum the per-thread gradients in parallel over parameter ranges
            let chunk = PARAMS.div_ceil(threads);
            thread::scope(|scope| {
                for (index, out) in total.chunks_mut(chunk).enumerate() {
                    let grads = &grads;
                    scope.spawn(move || {
                        let start = index * chunk;
                        let len = out.len();
                        out.iter_mut().for_each(|o| *o = 0.0);
                        for grad in grads {
                            for (o, g) in out.iter_mut().zip(&grad[start..start + len]) {
                                *o += *g;
                            }
                        }
                    });
                }
            });

            step += 1;
            let correction1 = 1.0 - beta1.powi(step);
            let correction2 = 1.0 - beta2.powi(step);
            let scale = 1.0 / BATCH as f32;
            for i in 0..PARAMS {
                let g = total[i] * scale;
                momentum[i] = beta1 * momentum[i] + (1.0 - beta1) * g;
                velocity[i] = beta2 * velocity[i] + (1.0 - beta2) * g * g;
                let update = (momentum[i] / correction1)
                    / ((velocity[i] / correction2).sqrt() + epsilon);
                params[i] -= lr * update;
                if i < B1 || (W2..B2).contains(&i) {
                    params[i] = params[i].clamp(-CLIP, CLIP);
                }
            }
        }
        let train_loss = epoch_loss / (batches_per_epoch * BATCH).max(1) as f64;
        let val_loss = validation_loss(&params, validation, &mut grads);
        eprintln!(
            "epoch {} train {train_loss:.6} validation {val_loss:.6} ({:.0}s)",
            epoch + 1,
            timer.elapsed().as_secs_f64()
        );
        save(&params, out_path);
    }
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
        "4k3/8/8/8/8/8/4PPP1/3RK3 w - - 0 1",
    ] {
        let board = crate::move_generation::Board::from_fen(fen);
        let position = PackedPosition::pack(&board, 0);
        let float = float_eval(&params, &position);
        let quantized = super::nnue::Accumulator::from_board(&board)
            .evaluate(board.white_to_move(), board.all_pieces().count_ones());
        eprintln!("{fen}: float {float:.1} quantized {quantized}");
    }
}

/// The unquantized network's evaluation in centipawns, side to move's view.
fn float_eval(params: &[f32], position: &PackedPosition) -> f32 {
    let stm = position.side_to_move as usize;
    let views = position_views(position, stm);
    let mut features = [Vec::new(), Vec::new()];
    position.for_each_piece(|color, piece, square| {
        features[0].push(views[0].feature(stm, color, piece, square));
        features[1].push(views[1].feature(1 - stm, color, piece, square));
    });
    let bucket = output_bucket(features[0].len() as u32);
    let mut out = params[B2 + bucket];
    for side in 0..2 {
        for j in 0..HIDDEN {
            let mut a = params[B1 + j];
            for &f in &features[side] {
                a += params[f * HIDDEN + j];
            }
            let c = a.clamp(0.0, 1.0);
            out += c * c * params[W2 + bucket * 2 * HIDDEN + side * HIDDEN + j];
        }
    }
    out * SCALE as f32
}

fn save(params: &[f32], path: &str) {
    let mut net: Box<Network> = unsafe { Box::new_zeroed().assume_init() };
    let quantize = |x: f32, q: i32| (x * q as f32).round().clamp(-32767.0, 32767.0) as i16;
    for f in 0..INPUTS {
        for j in 0..HIDDEN {
            net.feature_weights[f][j] = quantize(params[f * HIDDEN + j], QA);
        }
    }
    for j in 0..HIDDEN {
        net.feature_bias[j] = quantize(params[B1 + j], QA);
    }
    for b in 0..OUTPUT_BUCKETS {
        for j in 0..2 * HIDDEN {
            net.output_weights[b][j] = quantize(params[W2 + b * 2 * HIDDEN + j], QB);
        }
        net.output_bias[b] = quantize(params[B2 + b], QA * QB);
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(&*net as *const Network as *const u8, size_of::<Network>())
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("can't write {path}: {e}"));
}
