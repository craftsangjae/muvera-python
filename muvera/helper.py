"""Internal helper functions for MuVERA Fixed Dimensional Encoding.

This module contains low-level utilities for Gray code manipulation,
random projection matrix generation, Count Sketch, and SimHash-based
partition indexing. These are not part of the public API.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Gray code utilities
# ---------------------------------------------------------------------------


def append_to_gray_code(gray_code: int, bit: bool) -> int:
    """Append a single bit to a Gray code value.

    Parameters
    ----------
    gray_code : int
        Current Gray code value.
    bit : bool
        Bit to append (True=1, False=0).

    Returns
    -------
    int
        Updated Gray code.
    """
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))


def gray_code_to_binary(num: int) -> int:
    """Convert a Gray code value to its binary representation.

    Parameters
    ----------
    num : int
        Gray code value.

    Returns
    -------
    int
        Corresponding binary representation.
    """
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


# ---------------------------------------------------------------------------
# Random projection matrices
# ---------------------------------------------------------------------------


def simhash_matrix_from_seed(dimension: int, num_projections: int, seed: int) -> np.ndarray:
    """Generate a Gaussian random projection matrix for SimHash.

    Parameters
    ----------
    dimension : int
        Input vector dimension.
    num_projections : int
        Number of SimHash projections.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Float32 matrix of shape ``(dimension, num_projections)``.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(np.float32)


def ams_projection_matrix_from_seed(dimension: int, projection_dim: int, seed: int) -> np.ndarray:
    """Generate an AMS Sketch projection matrix.

    Each row has exactly one non-zero entry (+1 or -1), forming a sparse
    random projection.

    Parameters
    ----------
    dimension : int
        Input vector dimension.
    projection_dim : int
        Output (projected) dimension.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Float32 matrix of shape ``(dimension, projection_dim)``.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    indices = rng.integers(0, projection_dim, size=dimension)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
    out[np.arange(dimension), indices] = signs
    return out


def count_sketch_vector_from_seed(
    input_vector: np.ndarray, final_dimension: int, seed: int
) -> np.ndarray:
    """Project a vector to a lower dimension using Count Sketch.

    Parameters
    ----------
    input_vector : numpy.ndarray
        Input vector (1-D).
    final_dimension : int
        Output dimension.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Float32 vector of shape ``(final_dimension,)``.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(final_dimension, dtype=np.float32)
    indices = rng.integers(0, final_dimension, size=input_vector.shape[0])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_vector.shape[0])
    np.add.at(out, indices, signs * input_vector)
    return out


# ---------------------------------------------------------------------------
# Partition indexing
# ---------------------------------------------------------------------------


def _partition_index_gray_python(sketch_vector: np.ndarray) -> int:
    """Compute a Gray-code-based partition index (pure Python fallback)."""
    partition_index = 0
    for val in sketch_vector:
        partition_index = append_to_gray_code(partition_index, val > 0)
    return partition_index


def partition_index_gray(sketch_vector: np.ndarray) -> int:
    """Compute a Gray-code-based partition index from a SimHash sketch vector.

    Parameters
    ----------
    sketch_vector : numpy.ndarray
        SimHash projection result vector (1-D).

    Returns
    -------
    int
        Partition index.
    """
    from muvera import _RUST_AVAILABLE

    if _RUST_AVAILABLE:
        from muvera._rust_kernels import partition_index_gray as _rs_fn

        return int(_rs_fn(np.ascontiguousarray(sketch_vector, dtype=np.float32)))
    return _partition_index_gray_python(sketch_vector)


def distance_to_partition(sketch_vector: np.ndarray, partition_index: int) -> int:
    """Compute the Hamming distance between a sketch vector and a partition.

    Parameters
    ----------
    sketch_vector : numpy.ndarray
        SimHash projection result vector (1-D).
    partition_index : int
        Target partition index.

    Returns
    -------
    int
        Hamming distance.
    """
    num_projections = sketch_vector.size
    binary_representation = gray_code_to_binary(partition_index)
    sketch_bits = (sketch_vector > 0).astype(int)
    binary_array = (binary_representation >> np.arange(num_projections - 1, -1, -1)) & 1
    return int(np.sum(sketch_bits != binary_array))


# ---------------------------------------------------------------------------
# Vectorised batch helpers
# ---------------------------------------------------------------------------


def _partition_indices_gray_batch_python(sketches: np.ndarray) -> np.ndarray:
    """Compute Gray-code partition indices for a batch (pure Python fallback)."""
    num_projections = sketches.shape[1]
    bits = (sketches > 0).astype(np.uint32)
    partition_indices = np.zeros(sketches.shape[0], dtype=np.uint32)
    for bit_idx in range(num_projections):
        partition_indices = (partition_indices << 1) + (bits[:, bit_idx] ^ (partition_indices & 1))
    return partition_indices


def partition_indices_gray_batch(sketches: np.ndarray) -> np.ndarray:
    """Compute Gray-code partition indices for a batch of sketch vectors.

    Parameters
    ----------
    sketches : numpy.ndarray
        SimHash sketch matrix of shape ``(N, num_projections)``.

    Returns
    -------
    numpy.ndarray
        Uint32 partition index array of shape ``(N,)``.
    """
    from muvera import _RUST_AVAILABLE

    if _RUST_AVAILABLE:
        from muvera._rust_kernels import partition_indices_gray_batch as _rs_fn

        return np.asarray(
            _rs_fn(np.ascontiguousarray(sketches, dtype=np.float32)), dtype=np.uint32
        )
    return _partition_indices_gray_batch_python(sketches)
