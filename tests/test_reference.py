"""Tests verifying equivalence with the sionic-ai/muvera-py reference implementation.

The reference functions are inlined here to avoid an external dependency.
See: https://github.com/sionic-ai/muvera-py/blob/master/fde_generator.py
"""

import numpy as np
import pytest

from muvera import Muvera

# ---------------------------------------------------------------------------
# Reference implementation (inlined from sionic-ai/muvera-py)
# ---------------------------------------------------------------------------


def _ref_append_to_gray_code(gc, bit):
    return (gc << 1) + (int(bit) ^ (gc & 1))


def _ref_gray_code_to_binary(num):
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


def _ref_simhash_matrix(dim, nproj, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dim, nproj)).astype(np.float32)


def _ref_ams_matrix(dim, pdim, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((dim, pdim), dtype=np.float32)
    idx = rng.integers(0, pdim, size=dim)
    sgn = rng.choice([-1.0, 1.0], size=dim)
    out[np.arange(dim), idx] = sgn
    return out


def _ref_count_sketch(vec, fdim, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros(fdim, dtype=np.float32)
    idx = rng.integers(0, fdim, size=vec.shape[0])
    sgn = rng.choice([-1.0, 1.0], size=vec.shape[0])
    np.add.at(out, idx, sgn * vec)
    return out


def _ref_partition_gray(sv):
    pi = 0
    for v in sv:
        pi = _ref_append_to_gray_code(pi, v > 0)
    return pi


def _ref_dist_partition(sv, pi):
    n = sv.size
    br = _ref_gray_code_to_binary(pi)
    sb = (sv > 0).astype(int)
    ba = (br >> np.arange(n - 1, -1, -1)) & 1
    return int(np.sum(sb != ba))


def _ref_generate_fde(pc, dim, nreps, ksim, seed, encoding_avg, fill_empty, proj_type, pdim):
    n = pc.shape[0]
    npart = 2**ksim
    use_id = proj_type == "identity"
    actual_pdim = dim if use_id else pdim
    fde_dim = nreps * npart * actual_pdim
    out = np.zeros(fde_dim, dtype=np.float32)

    for rep in range(nreps):
        cs = seed + rep
        sk = pc @ _ref_simhash_matrix(dim, ksim, cs)
        proj = pc if use_id else pc @ _ref_ams_matrix(dim, actual_pdim, cs)

        rep_fde = np.zeros(npart * actual_pdim, dtype=np.float32)
        pcnt = np.zeros(npart, dtype=np.int32)
        for i in range(n):
            pi = _ref_partition_gray(sk[i])
            s = pi * actual_pdim
            rep_fde[s : s + actual_pdim] += proj[i]
            pcnt[pi] += 1

        if encoding_avg:
            for pi in range(npart):
                s = pi * actual_pdim
                if pcnt[pi] > 0:
                    rep_fde[s : s + actual_pdim] /= pcnt[pi]
                elif fill_empty and n > 0:
                    dists = [_ref_dist_partition(sk[j], pi) for j in range(n)]
                    rep_fde[s : s + actual_pdim] = proj[np.argmin(dists)]

        rs = rep * npart * actual_pdim
        out[rs : rs + rep_fde.size] = rep_fde
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def doc():
    return np.random.default_rng(123).standard_normal((80, 128)).astype(np.float32)


@pytest.fixture
def query():
    return np.random.default_rng(456).standard_normal((32, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReferenceEquivalence:
    """Verify our Muvera class produces identical output to the reference."""

    @pytest.mark.parametrize("nreps", [1, 2, 5])
    @pytest.mark.parametrize("ksim", [3, 4, 5])
    def test_query_fde_identity(self, query, nreps, ksim):
        enc = Muvera(
            num_repetitions=nreps,
            num_simhash_projections=ksim,
            dimension=128,
            seed=42,
            fill_empty_partitions=False,
        )
        ours = enc.encode_queries(query)
        ref = _ref_generate_fde(
            query,
            128,
            nreps,
            ksim,
            42,
            encoding_avg=False,
            fill_empty=False,
            proj_type="identity",
            pdim=None,
        )
        np.testing.assert_array_equal(ours, ref)

    @pytest.mark.parametrize("fill", [False, True])
    def test_document_fde_identity(self, doc, fill):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            seed=42,
            fill_empty_partitions=fill,
        )
        ours = enc.encode_documents(doc)
        ref = _ref_generate_fde(
            doc,
            128,
            2,
            4,
            42,
            encoding_avg=True,
            fill_empty=fill,
            proj_type="identity",
            pdim=None,
        )
        np.testing.assert_array_equal(ours, ref)

    def test_query_fde_ams(self, query):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            projection_type="ams_sketch",
            projection_dimension=16,
            seed=42,
            fill_empty_partitions=False,
        )
        ours = enc.encode_queries(query)
        ref = _ref_generate_fde(
            query,
            128,
            2,
            4,
            42,
            encoding_avg=False,
            fill_empty=False,
            proj_type="ams_sketch",
            pdim=16,
        )
        np.testing.assert_array_equal(ours, ref)

    def test_document_fde_ams_with_fill(self, doc):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            projection_type="ams_sketch",
            projection_dimension=16,
            seed=42,
            fill_empty_partitions=True,
        )
        ours = enc.encode_documents(doc)
        ref = _ref_generate_fde(
            doc,
            128,
            2,
            4,
            42,
            encoding_avg=True,
            fill_empty=True,
            proj_type="ams_sketch",
            pdim=16,
        )
        np.testing.assert_array_equal(ours, ref)

    def test_similarity_matches_reference(self, doc, query):
        enc = Muvera(
            num_repetitions=3,
            num_simhash_projections=4,
            dimension=128,
            seed=42,
            fill_empty_partitions=True,
        )
        our_q = enc.encode_queries(query)
        our_d = enc.encode_documents(doc)

        ref_q = _ref_generate_fde(
            query,
            128,
            3,
            4,
            42,
            encoding_avg=False,
            fill_empty=False,
            proj_type="identity",
            pdim=None,
        )
        ref_d = _ref_generate_fde(
            doc,
            128,
            3,
            4,
            42,
            encoding_avg=True,
            fill_empty=True,
            proj_type="identity",
            pdim=None,
        )
        np.testing.assert_allclose(np.dot(our_q, our_d), np.dot(ref_q, ref_d))

    def test_final_projection_matches_reference(self, doc):
        enc = Muvera(
            num_repetitions=2,
            num_simhash_projections=4,
            dimension=128,
            final_projection_dimension=1024,
            seed=42,
            fill_empty_partitions=True,
        )
        ours = enc.encode_documents(doc)

        # Generate without final projection, then apply count sketch
        ref_full = _ref_generate_fde(
            doc,
            128,
            2,
            4,
            42,
            encoding_avg=True,
            fill_empty=True,
            proj_type="identity",
            pdim=None,
        )
        ref = _ref_count_sketch(ref_full, 1024, 42)
        np.testing.assert_array_equal(ours, ref)
