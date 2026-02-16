"""Tests verifying Rust kernels produce identical results to Python implementations."""

from __future__ import annotations

import numpy as np
import pytest

from muvera import _RUST_AVAILABLE, Muvera
from muvera.helper import (
    _partition_index_gray_python,
    _partition_indices_gray_batch_python,
    simhash_matrix_from_seed,
)

pytestmark = pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust kernels not available")


class TestPartitionIndexGray:
    def test_single_matches_python(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            sketch = rng.standard_normal(5).astype(np.float32)
            py_result = _partition_index_gray_python(sketch)
            from muvera._rust_kernels import partition_index_gray as rs_fn

            rs_result = int(rs_fn(sketch))
            assert py_result == rs_result

    def test_batch_matches_python(self):
        rng = np.random.default_rng(42)
        sketches = rng.standard_normal((200, 5)).astype(np.float32)

        py_result = _partition_indices_gray_batch_python(sketches)
        from muvera._rust_kernels import partition_indices_gray_batch as rs_fn

        rs_result = np.asarray(rs_fn(sketches), dtype=np.uint32)
        np.testing.assert_array_equal(py_result, rs_result)

    def test_various_projection_counts(self):
        rng = np.random.default_rng(123)
        from muvera._rust_kernels import partition_index_gray as rs_fn

        for num_proj in range(1, 10):
            sketch = rng.standard_normal(num_proj).astype(np.float32)
            py_result = _partition_index_gray_python(sketch)
            rs_result = int(rs_fn(sketch))
            assert py_result == rs_result


class TestAggregateSingle:
    def test_query_aggregate_matches(self):
        from muvera._rust_kernels import aggregate_single as rs_fn

        rng = np.random.default_rng(42)
        dim = 16
        num_vecs = 20
        num_simhash = 3
        num_partitions = 2**num_simhash
        proj_dim = dim

        vectors = rng.standard_normal((num_vecs, dim)).astype(np.float32)
        sim_matrix = simhash_matrix_from_seed(dim, num_simhash, seed=42)
        sketches = (vectors @ sim_matrix).astype(np.float32)
        projected = vectors.copy()

        encoder = Muvera(
            num_repetitions=1,
            num_simhash_projections=num_simhash,
            dimension=dim,
            seed=42,
        )
        py_result = encoder._aggregate_single_python(sketches, projected, is_query=True)
        rs_result = np.asarray(
            rs_fn(sketches, projected, num_partitions, proj_dim, True, False, num_simhash),
            dtype=np.float32,
        )
        np.testing.assert_allclose(py_result, rs_result, atol=1e-6)

    def test_document_aggregate_with_fill_matches(self):
        from muvera._rust_kernels import aggregate_single as rs_fn

        rng = np.random.default_rng(42)
        dim = 16
        num_vecs = 5  # Few vectors to ensure empty partitions
        num_simhash = 5
        num_partitions = 2**num_simhash
        proj_dim = dim

        vectors = rng.standard_normal((num_vecs, dim)).astype(np.float32)
        sim_matrix = simhash_matrix_from_seed(dim, num_simhash, seed=42)
        sketches = (vectors @ sim_matrix).astype(np.float32)
        projected = vectors.copy()

        encoder = Muvera(
            num_repetitions=1,
            num_simhash_projections=num_simhash,
            dimension=dim,
            fill_empty_partitions=True,
            seed=42,
        )
        py_result = encoder._aggregate_single_python(sketches, projected, is_query=False)
        rs_result = np.asarray(
            rs_fn(sketches, projected, num_partitions, proj_dim, False, True, num_simhash),
            dtype=np.float32,
        )
        np.testing.assert_allclose(py_result, rs_result, atol=1e-6)


class TestScatterAdd:
    def test_scatter_add_matches(self):
        from muvera._rust_kernels import scatter_add_partitions as rs_fn

        rng = np.random.default_rng(42)
        batch_size = 3
        num_partitions = 8
        proj_dim = 4
        n = 50

        doc_indices = rng.integers(0, batch_size, size=n).astype(np.uint32)
        part_indices = rng.integers(0, num_partitions, size=n).astype(np.uint32)
        projected = rng.standard_normal((n, proj_dim)).astype(np.float32)

        # Python path
        py_fde = np.zeros((batch_size, num_partitions, proj_dim), dtype=np.float32)
        doc_part = doc_indices * num_partitions + part_indices
        base = doc_part * proj_dim
        flat = py_fde.reshape(-1)
        for d in range(proj_dim):
            np.add.at(flat, base + d, projected[:, d])

        # Rust path
        rs_fde = np.zeros((batch_size, num_partitions, proj_dim), dtype=np.float32)
        rs_fn(rs_fde, doc_indices, part_indices, projected)

        np.testing.assert_allclose(py_fde, rs_fde, atol=1e-6)


class TestEndToEnd:
    @pytest.mark.parametrize("num_simhash", [3, 5])
    @pytest.mark.parametrize("fill_empty", [True, False])
    def test_encode_documents_deterministic(self, num_simhash, fill_empty):
        rng = np.random.default_rng(42)
        dim = 32
        encoder = Muvera(
            num_repetitions=5,
            num_simhash_projections=num_simhash,
            dimension=dim,
            fill_empty_partitions=fill_empty,
            seed=42,
        )

        doc = rng.standard_normal((20, dim)).astype(np.float32)
        result1 = encoder.encode_documents(doc)
        result2 = encoder.encode_documents(doc)
        np.testing.assert_array_equal(result1, result2)

    def test_encode_batch_consistency(self):
        rng = np.random.default_rng(42)
        dim = 32
        encoder = Muvera(
            num_repetitions=5,
            num_simhash_projections=4,
            dimension=dim,
            seed=42,
        )

        docs = [
            rng.standard_normal((20, dim)).astype(np.float32),
            rng.standard_normal((15, dim)).astype(np.float32),
            rng.standard_normal((30, dim)).astype(np.float32),
        ]

        # Single encoding matches batch
        batch_result = encoder.encode_documents(docs)
        for i, doc in enumerate(docs):
            single_result = encoder.encode_documents(doc)
            np.testing.assert_allclose(batch_result[i], single_result, atol=1e-5)

    def test_encode_queries_batch_consistency(self):
        rng = np.random.default_rng(42)
        dim = 32
        encoder = Muvera(
            num_repetitions=5,
            num_simhash_projections=4,
            dimension=dim,
            seed=42,
        )

        queries = [
            rng.standard_normal((10, dim)).astype(np.float32),
            rng.standard_normal((8, dim)).astype(np.float32),
        ]

        batch_result = encoder.encode_queries(queries)
        for i, q in enumerate(queries):
            single_result = encoder.encode_queries(q)
            np.testing.assert_allclose(batch_result[i], single_result, atol=1e-5)

    def test_ams_sketch_mode(self):
        rng = np.random.default_rng(42)
        dim = 64
        encoder = Muvera(
            num_repetitions=5,
            num_simhash_projections=4,
            dimension=dim,
            projection_type="ams_sketch",
            projection_dimension=8,
            seed=42,
        )

        doc = rng.standard_normal((30, dim)).astype(np.float32)
        result1 = encoder.encode_documents(doc)
        result2 = encoder.encode_documents(doc)
        np.testing.assert_array_equal(result1, result2)
