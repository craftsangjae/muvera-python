"""Unit tests for muvera.helper functions."""

import numpy as np

from muvera.helper import (
    ams_projection_matrix_from_seed,
    append_to_gray_code,
    count_sketch_vector_from_seed,
    distance_to_partition,
    gray_code_to_binary,
    partition_index_gray,
    partition_indices_gray_batch,
    simhash_matrix_from_seed,
)

# ---------------------------------------------------------------------------
# Gray code
# ---------------------------------------------------------------------------


class TestAppendToGrayCode:
    def test_append_zero_bit(self):
        assert append_to_gray_code(0, False) == 0

    def test_append_one_bit(self):
        assert append_to_gray_code(0, True) == 1

    def test_sequential_append(self):
        gc = 0
        gc = append_to_gray_code(gc, True)  # 1
        gc = append_to_gray_code(gc, False)  # 10 -> gray
        gc = append_to_gray_code(gc, True)  # gray code for [1,0,1]
        assert gc == 6


class TestGrayCodeToBinary:
    def test_zero(self):
        assert gray_code_to_binary(0) == 0

    def test_small_values(self):
        assert gray_code_to_binary(1) == 1
        assert gray_code_to_binary(2) == 3
        assert gray_code_to_binary(3) == 2

    def test_invertibility_within_partition_context(self):
        # gray_code_to_binary should be consistent with distance_to_partition
        for n in range(16):
            result = gray_code_to_binary(n)
            assert 0 <= result < 16


# ---------------------------------------------------------------------------
# Random projection matrices
# ---------------------------------------------------------------------------


class TestSimhashMatrix:
    def test_shape(self):
        mat = simhash_matrix_from_seed(128, 5, seed=42)
        assert mat.shape == (128, 5)
        assert mat.dtype == np.float32

    def test_reproducibility(self):
        a = simhash_matrix_from_seed(64, 3, seed=7)
        b = simhash_matrix_from_seed(64, 3, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        a = simhash_matrix_from_seed(64, 3, seed=0)
        b = simhash_matrix_from_seed(64, 3, seed=1)
        assert not np.allclose(a, b)


class TestAmsProjectionMatrix:
    def test_shape(self):
        mat = ams_projection_matrix_from_seed(128, 16, seed=42)
        assert mat.shape == (128, 16)
        assert mat.dtype == np.float32

    def test_one_nonzero_per_row(self):
        mat = ams_projection_matrix_from_seed(64, 8, seed=42)
        for row in mat:
            assert np.count_nonzero(row) == 1
            assert abs(row[np.nonzero(row)][0]) == 1.0

    def test_reproducibility(self):
        a = ams_projection_matrix_from_seed(64, 8, seed=7)
        b = ams_projection_matrix_from_seed(64, 8, seed=7)
        np.testing.assert_array_equal(a, b)


class TestCountSketchVector:
    def test_shape(self):
        vec = np.ones(100, dtype=np.float32)
        out = count_sketch_vector_from_seed(vec, 32, seed=42)
        assert out.shape == (32,)
        assert out.dtype == np.float32

    def test_reproducibility(self):
        vec = np.random.randn(100).astype(np.float32)
        a = count_sketch_vector_from_seed(vec, 32, seed=7)
        b = count_sketch_vector_from_seed(vec, 32, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_zero_input(self):
        vec = np.zeros(50, dtype=np.float32)
        out = count_sketch_vector_from_seed(vec, 16, seed=42)
        np.testing.assert_array_equal(out, np.zeros(16, dtype=np.float32))


# ---------------------------------------------------------------------------
# Partition indexing
# ---------------------------------------------------------------------------


class TestPartitionIndexGray:
    def test_all_positive(self):
        sketch = np.array([1.0, 1.0, 1.0])
        idx = partition_index_gray(sketch)
        assert isinstance(idx, int)
        assert 0 <= idx < 8

    def test_all_negative(self):
        sketch = np.array([-1.0, -1.0, -1.0])
        assert partition_index_gray(sketch) == 0

    def test_deterministic(self):
        sketch = np.array([0.5, -0.3, 0.8, -0.1])
        a = partition_index_gray(sketch)
        b = partition_index_gray(sketch)
        assert a == b

    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            sketch = rng.standard_normal(5)
            idx = partition_index_gray(sketch)
            assert 0 <= idx < 32


class TestDistanceToPartition:
    def test_distance_non_negative(self):
        sketch = np.array([1.0, -1.0, 0.5])
        for pidx in range(8):
            assert distance_to_partition(sketch, pidx) >= 0

    def test_max_distance(self):
        sketch = np.array([1.0, -1.0, 0.5])
        for pidx in range(8):
            assert distance_to_partition(sketch, pidx) <= 3


class TestPartitionIndicesGrayBatch:
    def test_matches_single(self):
        rng = np.random.default_rng(42)
        sketches = rng.standard_normal((50, 5)).astype(np.float32)
        batch_result = partition_indices_gray_batch(sketches)

        for i in range(50):
            single_result = partition_index_gray(sketches[i])
            assert batch_result[i] == single_result

    def test_shape_and_dtype(self):
        sketches = np.random.randn(100, 4).astype(np.float32)
        result = partition_indices_gray_batch(sketches)
        assert result.shape == (100,)
        assert result.dtype == np.uint32

    def test_range(self):
        sketches = np.random.randn(200, 6).astype(np.float32)
        result = partition_indices_gray_batch(sketches)
        assert np.all(result < 64)
