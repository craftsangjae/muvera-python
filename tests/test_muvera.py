"""Integration tests for the Muvera class."""

import numpy as np
import pytest

from muvera import Muvera

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder():
    return Muvera(
        num_repetitions=2,
        num_simhash_projections=4,
        dimension=128,
        seed=42,
    )


@pytest.fixture
def doc_single():
    return np.random.default_rng(0).standard_normal((80, 128)).astype(np.float32)


@pytest.fixture
def doc_batch():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((80, 128)).astype(np.float32) for _ in range(5)]


@pytest.fixture
def query_single():
    return np.random.default_rng(1).standard_normal((32, 128)).astype(np.float32)


@pytest.fixture
def query_batch():
    rng = np.random.default_rng(1)
    return [rng.standard_normal((32, 128)).astype(np.float32) for _ in range(3)]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_output_dimension_identity(self):
        enc = Muvera(num_repetitions=10, num_simhash_projections=4, dimension=128)
        assert enc.output_dimension == 10 * 16 * 128

    def test_output_dimension_ams(self):
        enc = Muvera(
            num_repetitions=10,
            num_simhash_projections=4,
            dimension=128,
            projection_type="ams_sketch",
            projection_dimension=16,
        )
        assert enc.output_dimension == 10 * 16 * 16

    def test_output_dimension_final_projection(self):
        enc = Muvera(
            num_repetitions=10,
            num_simhash_projections=4,
            dimension=128,
            final_projection_dimension=1024,
        )
        assert enc.output_dimension == 1024

    def test_invalid_num_repetitions(self):
        with pytest.raises(ValueError, match="num_repetitions"):
            Muvera(num_repetitions=0)

    def test_invalid_simhash_projections(self):
        with pytest.raises(ValueError, match="num_simhash_projections"):
            Muvera(num_simhash_projections=31)

    def test_invalid_projection_type(self):
        with pytest.raises(ValueError, match="projection_type"):
            Muvera(projection_type="invalid")

    def test_ams_requires_projection_dimension(self):
        with pytest.raises(ValueError, match="projection_dimension"):
            Muvera(projection_type="ams_sketch")

    def test_repr(self):
        enc = Muvera(dimension=64, seed=7)
        r = repr(enc)
        assert "Muvera(" in r
        assert "dimension=64" in r
        assert "seed=7" in r


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestShapes:
    def test_single_document_shape(self, encoder, doc_single):
        fde = encoder.encode_documents(doc_single)
        assert fde.shape == (encoder.output_dimension,)
        assert fde.dtype == np.float32

    def test_batch_document_shape(self, encoder, doc_batch):
        fdes = encoder.encode_documents(doc_batch)
        assert fdes.shape == (5, encoder.output_dimension)
        assert fdes.dtype == np.float32

    def test_single_query_shape(self, encoder, query_single):
        fde = encoder.encode_queries(query_single)
        assert fde.shape == (encoder.output_dimension,)

    def test_batch_query_shape(self, encoder, query_batch):
        fdes = encoder.encode_queries(query_batch)
        assert fdes.shape == (3, encoder.output_dimension)

    def test_ams_sketch_shape(self, doc_single):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            projection_type="ams_sketch",
            projection_dimension=16,
            seed=42,
        )
        fde = enc.encode_documents(doc_single)
        assert fde.shape == (2 * 16 * 16,)

    def test_final_projection_shape(self, doc_single):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            final_projection_dimension=1024,
            seed=42,
        )
        fde = enc.encode_documents(doc_single)
        assert fde.shape == (1024,)

    def test_similarity_matrix_shape(self, encoder, doc_batch, query_batch):
        doc_fdes = encoder.encode_documents(doc_batch)
        query_fdes = encoder.encode_queries(query_batch)
        scores = query_fdes @ doc_fdes.T
        assert scores.shape == (3, 5)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_ndim(self, encoder):
        with pytest.raises(ValueError, match="Expected shape"):
            encoder.encode_documents(np.zeros(128, dtype=np.float32))

    def test_wrong_dimension_single(self, encoder):
        with pytest.raises(ValueError, match="Expected shape"):
            encoder.encode_documents(np.zeros((10, 64), dtype=np.float32))

    def test_wrong_dimension_batch(self, encoder):
        bad_batch = [np.zeros((10, 64), dtype=np.float32) for _ in range(5)]
        with pytest.raises(ValueError, match="expected"):
            encoder.encode_documents(bad_batch)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, doc_single):
        enc1 = Muvera(num_repetitions=2, num_simhash_projections=4, dimension=128, seed=42)
        enc2 = Muvera(num_repetitions=2, num_simhash_projections=4, dimension=128, seed=42)
        fde1 = enc1.encode_documents(doc_single)
        fde2 = enc2.encode_documents(doc_single)
        np.testing.assert_array_equal(fde1, fde2)

    def test_different_seed_different_result(self, doc_single):
        enc1 = Muvera(num_repetitions=2, num_simhash_projections=4, dimension=128, seed=42)
        enc2 = Muvera(num_repetitions=2, num_simhash_projections=4, dimension=128, seed=99)
        fde1 = enc1.encode_documents(doc_single)
        fde2 = enc2.encode_documents(doc_single)
        assert not np.allclose(fde1, fde2)

    def test_multiple_calls_same_result(self, encoder, doc_single):
        fde1 = encoder.encode_documents(doc_single)
        fde2 = encoder.encode_documents(doc_single)
        np.testing.assert_array_equal(fde1, fde2)


# ---------------------------------------------------------------------------
# Single vs batch consistency
# ---------------------------------------------------------------------------


class TestSingleBatchConsistency:
    def test_document_single_vs_batch(self, encoder, doc_batch):
        batch_fdes = encoder.encode_documents(doc_batch)
        for i in range(len(doc_batch)):
            single_fde = encoder.encode_documents(doc_batch[i])
            np.testing.assert_allclose(single_fde, batch_fdes[i], rtol=1e-4, atol=1e-6)

    def test_query_single_vs_batch(self, encoder, query_batch):
        batch_fdes = encoder.encode_queries(query_batch)
        for i in range(len(query_batch)):
            single_fde = encoder.encode_queries(query_batch[i])
            np.testing.assert_allclose(single_fde, batch_fdes[i], rtol=1e-4, atol=1e-6)

    def test_document_single_vs_batch_with_fill(self, doc_batch):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            fill_empty_partitions=True,
            seed=42,
        )
        batch_fdes = enc.encode_documents(doc_batch)
        for i in range(len(doc_batch)):
            single_fde = enc.encode_documents(doc_batch[i])
            np.testing.assert_allclose(single_fde, batch_fdes[i], rtol=1e-4, atol=1e-6)

    def test_document_single_vs_batch_ams(self, doc_batch):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            projection_type="ams_sketch",
            projection_dimension=16,
            seed=42,
        )
        batch_fdes = enc.encode_documents(doc_batch)
        for i in range(len(doc_batch)):
            single_fde = enc.encode_documents(doc_batch[i])
            np.testing.assert_allclose(single_fde, batch_fdes[i], rtol=1e-4, atol=1e-6)
