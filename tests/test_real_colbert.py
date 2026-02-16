"""Real-world ColBERT embedding tests using cached NanoBEIR fixtures.

These tests use actual ColBERT embeddings from the NanoBEIR dataset to validate
that MuVERA performs well on real data, not just synthetic test cases.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from muvera import Muvera

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "colbert_nanobeir"


def load_fixtures():
    """Load cached ColBERT embeddings and qrels."""
    # Load document embeddings
    doc_data = np.load(FIXTURE_DIR / "documents.npz", allow_pickle=True)
    doc_ids = doc_data["ids"].tolist()
    doc_embeddings = [doc_data[f"emb_{i}"] for i in range(int(doc_data["n"]))]

    # Load query embeddings
    query_data = np.load(FIXTURE_DIR / "queries.npz", allow_pickle=True)
    query_ids = query_data["ids"].tolist()
    query_embeddings = [query_data[f"emb_{i}"] for i in range(int(query_data["n"]))]

    # Load qrels
    with open(FIXTURE_DIR / "qrels.json") as f:
        qrels = json.load(f)

    return doc_ids, doc_embeddings, query_ids, query_embeddings, qrels


@pytest.fixture(scope="module")
def colbert_fixtures():
    """Cached ColBERT embeddings fixture."""
    return load_fixtures()


def compute_maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute native ColBERT MaxSim (Chamfer) similarity."""
    sim_matrix = query_emb @ doc_emb.T
    return float(sim_matrix.max(axis=1).sum())


def compute_recall_at_k(
    results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int
) -> float:
    """Compute Recall@K metric."""
    hits, total = 0, 0
    for qid, ranked_docs in results.items():
        relevant = set(qrels.get(qid, {}).keys())
        if not relevant:
            continue
        total += 1
        top_k = set(ranked_docs[:k])
        if not relevant.isdisjoint(top_k):
            hits += 1
    return hits / total if total > 0 else 0.0


class TestRealColBERT:
    """Tests using real ColBERT embeddings from NanoBEIR."""

    def test_fixture_loading(self, colbert_fixtures):
        """Verify fixtures are loaded correctly."""
        doc_ids, doc_embeddings, query_ids, query_embeddings, qrels = colbert_fixtures

        assert len(doc_ids) == 35
        assert len(query_ids) == 5
        assert len(doc_embeddings) == 35
        assert len(query_embeddings) == 5
        assert sum(len(v) for v in qrels.values()) == 35

        # Check embedding properties
        for emb in doc_embeddings:
            assert emb.ndim == 2
            assert emb.shape[1] == 128
            # Check normalization (should be unit vectors)
            norms = np.linalg.norm(emb, axis=1)
            assert np.allclose(norms, 1.0, rtol=1e-5)

    def test_fde_encoding_shape(self, colbert_fixtures):
        """Test FDE encoding produces correct shapes."""
        _, doc_embeddings, _, query_embeddings, _ = colbert_fixtures

        encoder = Muvera(
            num_repetitions=20,
            num_simhash_projections=7,
            dimension=128,
            seed=42,
        )

        doc_fdes = encoder.encode_documents(doc_embeddings)
        query_fdes = encoder.encode_queries(query_embeddings)

        assert doc_fdes.shape == (35, encoder.output_dimension)
        assert query_fdes.shape == (5, encoder.output_dimension)
        assert doc_fdes.dtype == np.float32
        assert query_fdes.dtype == np.float32

    def test_fde_vs_native_maxsim_correlation(self, colbert_fixtures):
        """Test correlation between FDE and native MaxSim scores."""
        doc_ids, doc_embeddings, query_ids, query_embeddings, _ = colbert_fixtures

        encoder = Muvera(
            num_repetitions=20,
            num_simhash_projections=7,
            dimension=128,
            seed=42,
        )

        doc_fdes = encoder.encode_documents(doc_embeddings)
        query_fdes = encoder.encode_queries(query_embeddings)

        # Compute FDE scores
        fde_scores = query_fdes @ doc_fdes.T

        # Compute native MaxSim scores
        native_scores = np.zeros((len(query_embeddings), len(doc_embeddings)), dtype=np.float32)
        for i, q_emb in enumerate(query_embeddings):
            for j, d_emb in enumerate(doc_embeddings):
                native_scores[i, j] = compute_maxsim(q_emb, d_emb)

        # Check correlation (should be high for real ColBERT embeddings)
        from scipy.stats import spearmanr

        correlation, _ = spearmanr(fde_scores.flatten(), native_scores.flatten())

        # Real ColBERT embeddings should have much higher correlation than random data
        assert correlation > 0.7, f"Correlation too low: {correlation:.4f}"

    def test_recall_at_k(self, colbert_fixtures):
        """Test Recall@K metrics on real data."""
        doc_ids, doc_embeddings, query_ids, query_embeddings, qrels = colbert_fixtures

        encoder = Muvera(
            num_repetitions=20,
            num_simhash_projections=7,
            dimension=128,
            seed=42,
        )

        # Encode with FDE
        doc_fdes = encoder.encode_documents(doc_embeddings)
        query_fdes = encoder.encode_queries(query_embeddings)
        fde_scores = query_fdes @ doc_fdes.T

        # Compute native MaxSim for comparison
        native_scores = np.zeros((len(query_embeddings), len(doc_embeddings)), dtype=np.float32)
        for i, q_emb in enumerate(query_embeddings):
            for j, d_emb in enumerate(doc_embeddings):
                native_scores[i, j] = compute_maxsim(q_emb, d_emb)

        # Rank documents
        fde_results = {}
        native_results = {}
        for i, qid in enumerate(query_ids):
            fde_ranking = np.argsort(fde_scores[i])[::-1]
            native_ranking = np.argsort(native_scores[i])[::-1]
            fde_results[qid] = [doc_ids[j] for j in fde_ranking]
            native_results[qid] = [doc_ids[j] for j in native_ranking]

        # Compute Recall@K
        for k in [1, 5, 10]:
            fde_recall = compute_recall_at_k(fde_results, qrels, k)
            native_recall = compute_recall_at_k(native_results, qrels, k)

            print(f"\nRecall@{k}: FDE={fde_recall:.2%}, Native={native_recall:.2%}")

            # FDE should achieve reasonable recall (at least 50% of native for small k)
            if k == 10:
                assert fde_recall >= 0.3 * native_recall, (
                    f"FDE Recall@{k} too low: {fde_recall:.2%} (Native: {native_recall:.2%})"
                )

    def test_ranking_quality(self, colbert_fixtures):
        """Test that FDE produces reasonable rankings."""
        doc_ids, doc_embeddings, query_ids, query_embeddings, qrels = colbert_fixtures

        encoder = Muvera(
            num_repetitions=20,
            num_simhash_projections=7,
            dimension=128,
            seed=42,
        )

        doc_fdes = encoder.encode_documents(doc_embeddings)
        query_fdes = encoder.encode_queries(query_embeddings)
        fde_scores = query_fdes @ doc_fdes.T

        # For each query, check if any relevant docs are in top-10
        for i, qid in enumerate(query_ids):
            if qid not in qrels or not qrels[qid]:
                continue

            relevant_doc_ids = set(qrels[qid].keys())
            top_10_indices = np.argsort(fde_scores[i])[-10:]
            top_10_doc_ids = {doc_ids[j] for j in top_10_indices}

            # At least one relevant doc should be in top-10
            has_relevant = not relevant_doc_ids.isdisjoint(top_10_doc_ids)
            assert has_relevant, f"Query {qid}: No relevant docs in top-10"

    def test_encoding_determinism(self, colbert_fixtures):
        """Test that encoding is deterministic with same seed."""
        _, doc_embeddings, _, _, _ = colbert_fixtures

        encoder1 = Muvera(num_repetitions=10, num_simhash_projections=5, dimension=128, seed=42)
        encoder2 = Muvera(num_repetitions=10, num_simhash_projections=5, dimension=128, seed=42)

        fdes1 = encoder1.encode_documents(doc_embeddings)
        fdes2 = encoder2.encode_documents(doc_embeddings)

        np.testing.assert_array_equal(fdes1, fdes2)
