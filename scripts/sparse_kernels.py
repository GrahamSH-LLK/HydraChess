"""Sparse feature-transformer kernels for the MLX trainer.

Each position has at most 32 active inputs out of 768, so instead of a dense matmul the forward
pass sums the active weight rows, and the backward pass sums, for every weight row, the gradients
of the positions where that input was active. The positions are grouped by input on the CPU
(`group_by_feature`), which keeps the backward kernel free of atomics and deterministic.
"""

import mlx.core as mx
import numpy as np

INPUTS = 768
SLOTS = 32

CHUNK = 128

_forward = mx.fast.metal_kernel(
    name="sparse_rows_sum4",
    input_names=["weights", "features"],
    output_names=["out"],
    source="""
        // Each thread sums four adjacent hidden units with 128-bit loads
        uint j4 = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        uint quads = weights_shape[1] / 4;
        const device float4* rows = (const device float4*)weights;
        float4 sum = float4(0.0f);
        for (int k = 0; k < 32; k++) {
            int f = features[b * 32 + k];
            if (f < 768) {
                sum += rows[f * quads + j4];
            }
        }
        ((device float4*)out)[b * quads + j4] = sum;
    """,
)

_partial = mx.fast.metal_kernel(
    name="sparse_chunk_grad4",
    input_names=["grad", "positions", "chunk_start", "chunk_end"],
    output_names=["out"],
    source="""
        // One chunk of at most CHUNK positions that share an input
        uint j4 = thread_position_in_grid.x;
        uint c = thread_position_in_grid.y;
        uint quads = grad_shape[1] / 4;
        const device float4* g = (const device float4*)grad;
        float4 sum = float4(0.0f);
        for (int i = chunk_start[c]; i < chunk_end[c]; i++) {
            sum += g[positions[i] * quads + j4];
        }
        ((device float4*)out)[c * quads + j4] = sum;
    """,
)

_reduce = mx.fast.metal_kernel(
    name="sparse_chunk_reduce4",
    input_names=["partial", "chunk_offsets"],
    output_names=["out"],
    source="""
        // Adds up the chunks of each input
        uint j4 = thread_position_in_grid.x;
        uint f = thread_position_in_grid.y;
        uint quads = partial_shape[1] / 4;
        const device float4* p = (const device float4*)partial;
        float4 sum = float4(0.0f);
        for (int c = chunk_offsets[f]; c < chunk_offsets[f + 1]; c++) {
            sum += p[c * quads + j4];
        }
        ((device float4*)out)[f * quads + j4] = sum;
    """,
)


def _threadgroup(quads):
    return (min(quads, 128), 1, 1)


def rows_sum(weights, features):
    batch = features.shape[0]
    hidden = weights.shape[1]
    return _forward(
        inputs=[weights, features],
        grid=(hidden // 4, batch, 1),
        threadgroup=_threadgroup(hidden // 4),
        output_shapes=[(batch, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def rows_grad(grad, positions, chunk_start, chunk_end, chunk_offsets):
    hidden = grad.shape[1]
    chunks = chunk_start.shape[0]
    partial = _partial(
        inputs=[grad, positions, chunk_start, chunk_end],
        grid=(hidden // 4, chunks, 1),
        threadgroup=_threadgroup(hidden // 4),
        output_shapes=[(chunks, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return _reduce(
        inputs=[partial, chunk_offsets],
        grid=(hidden // 4, INPUTS, 1),
        threadgroup=_threadgroup(hidden // 4),
        output_shapes=[(INPUTS, hidden)],
        output_dtypes=[mx.float32],
    )[0]


@mx.custom_function
def sparse_affine(weights, features, positions, chunk_start, chunk_end, chunk_offsets):
    """sum of weights[f] over the active features f of each position: (batch, hidden).

    The other arguments come from `group_by_feature` and are only used for the gradient."""
    return rows_sum(weights, features)


@sparse_affine.vjp
def _sparse_affine_vjp(primals, cotangent, output):
    weights, features, positions, chunk_start, chunk_end, chunk_offsets = primals
    grad = rows_grad(cotangent, positions, chunk_start, chunk_end, chunk_offsets)
    return (grad,) + tuple(mx.zeros_like(x) for x in primals[1:])


def group_by_feature(features):
    """Positions sorted by the inputs active in them, split into chunks of at most CHUNK.

    `features` is a (batch, 32) int array padded with INPUTS. Returns int32 arrays: `positions`;
    `chunk_start` and `chunk_end`, ranges of `positions` that share one input; and
    `chunk_offsets` (INPUTS + 1 entries), so input f owns chunks chunk_offsets[f]..[f + 1].
    The padding input is left out.
    """
    flat = features.ravel()
    # A stable sort on int16 keys is a radix sort in numpy: linear time
    order = np.argsort(flat.astype(np.int16), kind="stable")
    positions = (order // SLOTS).astype(np.int32)
    counts = np.bincount(flat, minlength=INPUTS + 1)[:INPUTS]
    starts = np.zeros(INPUTS, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    per_input = (counts + CHUNK - 1) // CHUNK
    chunk_offsets = np.zeros(INPUTS + 1, dtype=np.int32)
    np.cumsum(per_input, out=chunk_offsets[1:])
    owner = np.repeat(np.arange(INPUTS), per_input)
    index_in_input = np.arange(len(owner)) - chunk_offsets[owner]
    chunk_start = starts[owner] + index_in_input * CHUNK
    chunk_end = np.minimum(chunk_start + CHUNK, starts[owner] + counts[owner])
    if len(owner) == 0:
        chunk_start = chunk_end = np.zeros(1, dtype=np.int64)
    return (
        positions,
        chunk_start.astype(np.int32),
        chunk_end.astype(np.int32),
        chunk_offsets,
    )
