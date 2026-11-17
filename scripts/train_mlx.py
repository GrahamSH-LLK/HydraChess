"""NNUE trainer on the Apple GPU with MLX. Same data, net and output format as `HydraChess train`.

    python train_mlx.py OUT.nnue EPOCHS DATA.bin [DATA.bin ...] [--hidden 512] [--batch 16384]

The net is (768 -> HIDDEN) x 2 perspectives -> SCReLU -> 8 output buckets by piece count, and
the output file is the raw bytes of `nnue::Network` (little-endian i16, padded to 64 bytes),
so `HIDDEN` in nnue.rs must match `--hidden`.
"""

import argparse
import math
import os
import queue
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from sparse_kernels import group_by_feature, sparse_affine

INPUTS = 768
OUTPUT_BUCKETS = 8
QA, QB, SCALE = 255, 64, 400
WDL = 0.3
CLIP = 1.98

RECORD = np.dtype(
    [
        ("occupancy", "<u8"),
        ("pieces", "u1", 16),
        ("score", "<i2"),
        ("result", "u1"),
        ("stm", "u1"),
        ("pad", "u1", 4),
    ]
)
assert RECORD.itemsize == 32


def load(paths):
    # Read straight into one array, so the files aren't held twice while concatenating
    sizes = [os.path.getsize(path) // 32 for path in paths]
    out = np.empty(sum(sizes), dtype=RECORD)
    raw = out.view(np.uint8)
    offset = 0
    for path, size in zip(paths, sizes):
        with open(path, "rb") as f:
            f.readinto(memoryview(raw[offset * 32 : (offset + size) * 32]))
        offset += size
    return out


def decode(batch):
    """Feature indices for the side to move and the other side, padded with INPUTS (a zero
    row), plus output buckets and targets."""
    n = len(batch)
    occ = batch["occupancy"].astype(np.uint64)
    bits = ((occ[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    # The i-th occupied square (ascending) holds the i-th nibble
    nibble_index = np.cumsum(bits, axis=1) - 1
    nibbles = np.empty((n, 32), dtype=np.uint8)
    nibbles[:, 0::2] = batch["pieces"] & 0xF
    nibbles[:, 1::2] = batch["pieces"] >> 4
    rows, squares = np.nonzero(bits)
    nib = nibbles[rows, nibble_index[rows, squares]]
    color = (nib >> 3).astype(np.int32)
    piece = (nib & 7).astype(np.int32)
    squares = squares.astype(np.int32)
    white_view = (color != 0) * 384 + piece * 64 + squares
    black_view = (color != 1) * 384 + piece * 64 + (squares ^ 56)
    stm = batch["stm"].astype(bool)
    us_view = np.where(stm[rows], black_view, white_view)
    them_view = np.where(stm[rows], white_view, black_view)

    # Slot of each piece within its row
    counts = bits.sum(axis=1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(rows)) - starts[rows]
    us = np.full((n, 32), INPUTS, dtype=np.int32)
    them = np.full((n, 32), INPUTS, dtype=np.int32)
    us[rows, slot] = us_view
    them[rows, slot] = them_view

    bucket = np.minimum((counts - 2) // 4, OUTPUT_BUCKETS - 1).astype(np.int32)
    score = batch["score"].astype(np.float32)
    result = batch["result"].astype(np.float32) / 2
    score = np.where(stm, -score, score)
    result = np.where(stm, 1 - result, result)
    target = (1 - WDL) / (1 + np.exp(-score / SCALE)) + WDL * result
    return (us, them, bucket, target.astype(np.float32)) + group_by_feature(us) + group_by_feature(them)


class Net(nn.Module):
    def __init__(self, hidden, rng):
        super().__init__()
        self.hidden = hidden
        w1 = (rng.random((INPUTS, hidden), dtype=np.float32) * 2 - 1) * math.sqrt(1 / 32)
        w2 = (rng.random((OUTPUT_BUCKETS, 2 * hidden), dtype=np.float32) * 2 - 1) * math.sqrt(
            1 / (2 * hidden)
        )
        self.w1 = mx.array(w1)
        self.b1 = mx.zeros((hidden,))
        self.w2 = mx.array(w2)
        self.b2 = mx.zeros((OUTPUT_BUCKETS,))

    def accumulate(self, features, groups):
        # Sums the active weight rows with a custom kernel. A dense one-hot matmul is about as
        # fast here, but Apple's float32 matmul loses precision (errors around 1e-3)
        return sparse_affine(self.w1, features, *groups) + self.b1

    def __call__(self, us, them, bucket, *groups):
        a = mx.concatenate(
            [self.accumulate(us, groups[:4]), self.accumulate(them, groups[4:])], axis=1
        )
        a = mx.clip(a, 0, 1) ** 2
        return (a * self.w2[bucket]).sum(axis=1) + self.b2[bucket]


def loss_fn(model, us, them, bucket, target, *groups):
    predicted = mx.sigmoid(model(us, them, bucket, *groups))
    return ((predicted - target) ** 2).mean()


def save(model, path):
    def q(x, scale):
        return np.clip(np.round(np.array(x) * scale), -32767, 32767).astype("<i2")

    parts = [
        q(model.w1, QA).tobytes(),
        q(model.b1, QA).tobytes(),
        q(model.w2, QB).tobytes(),
        q(model.b2, QA * QB).tobytes(),
    ]
    data = b"".join(parts)
    data += b"\0" * (-len(data) % 64)
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("epochs", type=int)
    parser.add_argument("data", nargs="+")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=0.00002)
    parser.add_argument("--seed", type=int, default=12648430)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    positions = load(args.data)
    rng.shuffle(positions)
    validation_size = min(len(positions) // 100, 200_000)
    validation, training = positions[:validation_size], positions[validation_size:]
    print(f"{len(training)} training positions, {len(validation)} validation", flush=True)

    model = Net(args.hidden, rng)
    batches_per_epoch = len(training) // args.batch
    total_steps = max(1, args.epochs * batches_per_epoch)
    schedule = optim.cosine_decay(args.lr, total_steps, args.min_lr)
    optimizer = optim.Adam(learning_rate=schedule, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(*batch):
        loss, grads = loss_and_grad(model, *batch)
        optimizer.update(model, grads)
        model.w1 = mx.clip(model.w1, -CLIP, CLIP)
        model.w2 = mx.clip(model.w2, -CLIP, CLIP)
        return loss

    state = [model.state, optimizer.state]
    step = mx.compile(step, inputs=state, outputs=state)

    validation_batches = [
        tuple(mx.array(x) for x in decode(validation[i : i + args.batch]))
        for i in range(0, len(validation), args.batch)
    ]

    # Decode batches on CPU threads while the GPU trains on earlier ones
    orders = {}
    orders_lock = threading.Lock()

    def epoch_order(epoch):
        # One shared order per epoch; only the current and next epochs are kept
        with orders_lock:
            if epoch not in orders:
                rng = np.random.default_rng((args.seed, epoch))
                orders[epoch] = rng.permutation(len(training)).astype(np.uint32)
                for old in [e for e in orders if e < epoch - 1]:
                    del orders[old]
            return orders[epoch]

    def producer(out, worker, workers):
        for epoch in range(args.epochs):
            order = epoch_order(epoch)
            for b in range(worker, batches_per_epoch, workers):
                out[b % workers].put(decode(training[order[b * args.batch : (b + 1) * args.batch]]))

    workers = 4
    queues = [queue.Queue(maxsize=4) for _ in range(workers)]
    for w in range(workers):
        threading.Thread(target=producer, args=(queues, w, workers), daemon=True).start()

    timer = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        for b in range(batches_per_epoch):
            losses.append(step(*(mx.array(x) for x in queues[b % workers].get())))
            if len(losses) % 64 == 0:
                mx.eval(losses[-1])
        train_loss = float(mx.array(losses).mean().item())
        val_total = 0.0
        for batch in validation_batches:
            val_total += float(loss_fn(model, *batch).item()) * batch[3].shape[0]
        val_loss = val_total / len(validation)
        print(
            f"epoch {epoch + 1} train {train_loss:.6f} validation {val_loss:.6f} "
            f"({time.perf_counter() - timer:.0f}s)",
            flush=True,
        )
        save(model, args.out)


if __name__ == "__main__":
    main()
