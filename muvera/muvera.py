"""MuVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings).

This module provides the Fixed Dimensional Encoding (FDE) algorithm that
converts multi-vector embeddings (point clouds) into single fixed-dimensional
vectors.

References
----------
.. [1] Google graph-mining: fixed_dimensional_encoding.cc
   https://github.com/google/graph-mining/blob/main/sketching/point_cloud/fixed_dimensional_encoding.cc
.. [2] sionic-ai/muvera-py: fde_generator.py
   https://github.com/sionic-ai/muvera-py/blob/master/fde_generator.py
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from muvera.helper import (
    ams_projection_matrix,
    count_sketch_vector,
    distance_to_partition,
    gray_code_to_binary,
    partition_index_gray,
    partition_indices_gray_batch,
    simhash_matrix,
)


class Muvera:
    """Encoder that converts multi-vector embeddings into Fixed Dimensional Encodings.

    Uses the MuVERA algorithm to encode variable-length multi-vector
    representations (point clouds) into fixed-dimensional single vectors.
    The dot product between encoded vectors approximates the similarity
    between the original multi-vector sets.

    Parameters
    ----------
    num_repetitions : int, default=20
        Number of repetitions for FDE generation. Higher values improve
        accuracy but proportionally increase the output dimension.
    num_simhash_projections : int, default=5
        Number of SimHash projections. The number of partitions is
        ``2 ** num_simhash_projections``. Must be in ``[0, 31)``.
    dimension : int, default=16
        Dimension of the input embedding vectors.
    projection_type : {'identity', 'ams_sketch'}, default='identity'
        Inner projection method.

        - ``'identity'``: Uses the original dimension as-is.
        - ``'ams_sketch'``: Uses AMS Sketch for dimensionality reduction.
    projection_dimension : int or None, default=None
        Projected dimension when ``projection_type='ams_sketch'``.
        Ignored when ``'identity'``.
    fill_empty_partitions : bool, default=True
        Whether to fill empty partitions with the nearest vector during
        document encoding.
    final_projection_dimension : int or None, default=None
        Final Count Sketch projection dimension. ``None`` disables
        final projection.
    seed : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    output_dimension : int
        Dimension of the encoded output vector.

    Examples
    --------
    >>> import numpy as np
    >>> from muvera import Muvera
    >>> encoder = Muvera(num_repetitions=10, num_simhash_projections=4,
    ...                  dimension=128, seed=42)
    >>> docs = np.random.randn(5, 80, 128).astype(np.float32)   # 5 documents, 80 vectors each
    >>> queries = np.random.randn(3, 32, 128).astype(np.float32) # 3 queries, 32 vectors each
    >>> doc_fdes = encoder.encode_documents(docs)
    >>> query_fdes = encoder.encode_queries(queries)
    >>> scores = query_fdes @ doc_fdes.T  # (3, 5) similarity matrix
    """

    def __init__(
        self,
        num_repetitions: int = 20,
        num_simhash_projections: int = 5,
        dimension: int = 16,
        projection_type: Literal["identity", "ams_sketch"] = "identity",
        projection_dimension: int | None = None,
        fill_empty_partitions: bool = True,
        final_projection_dimension: int | None = None,
        seed: int = 42,
    ):
        if num_repetitions <= 0:
            raise ValueError(f"num_repetitions must be greater than 0, got {num_repetitions}")
        if not (0 <= num_simhash_projections < 31):
            raise ValueError(
                f"num_simhash_projections must be in [0, 31), got {num_simhash_projections}"
            )
        if projection_type not in ("identity", "ams_sketch"):
            raise ValueError(
                f"projection_type must be 'identity' or 'ams_sketch', got '{projection_type}'"
            )
        if projection_type == "ams_sketch" and (
            projection_dimension is None or projection_dimension <= 0
        ):
            raise ValueError(
                "A positive projection_dimension is required when projection_type='ams_sketch'"
            )

        self.num_repetitions = num_repetitions
        self.num_simhash_projections = num_simhash_projections
        self.dimension = dimension
        self.projection_type = projection_type
        self.projection_dimension = projection_dimension
        self.fill_empty_partitions = fill_empty_partitions
        self.final_projection_dimension = final_projection_dimension
        self.seed = seed

        # Derived constants
        self._num_partitions: int = 2**num_simhash_projections
        self._use_identity: bool = projection_type == "identity"
        self._proj_dim: int = dimension if self._use_identity else projection_dimension  # type: ignore[assignment]
        self._fde_dim: int = num_repetitions * self._num_partitions * self._proj_dim

        # Pre-generate per-repetition RNG seeds (deterministic from master seed).
        # Each repetition gets an independent seed to ensure reproducibility
        # regardless of call order.
        master_rng = np.random.default_rng(seed)
        self._rep_seeds: np.ndarray = master_rng.integers(
            0, 2**63, size=num_repetitions, dtype=np.int64
        )

        # Separate seed for the final projection
        self._final_proj_seed: int = int(master_rng.integers(0, 2**63))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dimension(self) -> int:
        """Dimension of the encoded output vector.

        Returns
        -------
        int
            ``final_projection_dimension`` if set, otherwise
            ``num_repetitions * num_partitions * projection_dim``.
        """
        if self.final_projection_dimension is not None and self.final_projection_dimension > 0:
            return self.final_projection_dimension
        return self._fde_dim

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def encode_documents(self, documents: np.ndarray) -> np.ndarray:
        """Encode document point clouds into Fixed Dimensional Encodings.

        Each document is a set of embedding vectors (point cloud) and is
        encoded using the AVERAGE method. When ``fill_empty_partitions=True``,
        empty partitions are filled with the nearest vector.

        Parameters
        ----------
        documents : numpy.ndarray
            Document embedding array. Supports the following shapes:

            - ``(num_vectors, dimension)`` : Single document.
            - ``(num_documents, num_vectors, dimension)`` : Batch of documents.

        Returns
        -------
        numpy.ndarray
            Encoded FDE vector(s).

            - Single document input: shape ``(output_dimension,)``.
            - Batch input: shape ``(num_documents, output_dimension)``.

        Raises
        ------
        ValueError
            If the input shape is invalid or dimension does not match.

        Notes
        -----
        Document encoding computes the centroid (average) of vectors within
        each partition, corresponding to the ``AVERAGE`` encoding type in the
        original C++ implementation.
        """
        return self._encode(documents, is_query=False)

    def encode_queries(self, queries: np.ndarray) -> np.ndarray:
        """Encode query point clouds into Fixed Dimensional Encodings.

        Each query is a set of embedding vectors (point cloud) and is
        encoded using the SUM method.

        Parameters
        ----------
        queries : numpy.ndarray
            Query embedding array. Supports the following shapes:

            - ``(num_vectors, dimension)`` : Single query.
            - ``(num_queries, num_vectors, dimension)`` : Batch of queries.

        Returns
        -------
        numpy.ndarray
            Encoded FDE vector(s).

            - Single query input: shape ``(output_dimension,)``.
            - Batch input: shape ``(num_queries, output_dimension)``.

        Raises
        ------
        ValueError
            If the input shape is invalid or dimension does not match.

        Notes
        -----
        Query encoding computes the sum of vectors within each partition,
        and ``fill_empty_partitions`` is ignored. This corresponds to the
        ``DEFAULT_SUM`` encoding type in the original C++ implementation.
        """
        return self._encode(queries, is_query=True)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _encode(self, data: np.ndarray, *, is_query: bool) -> np.ndarray:
        """Dispatch to single or batch encoding based on input dimensionality.

        Parameters
        ----------
        data : numpy.ndarray
            2-D ``(num_vectors, dimension)`` or 3-D ``(batch, num_vectors, dimension)``.
        is_query : bool
            If True, uses SUM encoding; if False, uses AVERAGE encoding.

        Returns
        -------
        numpy.ndarray
            Encoded result.
        """
        if data.ndim == 2:
            return self._encode_single(data, is_query=is_query)
        elif data.ndim == 3:
            return self._encode_batch(data, is_query=is_query)
        else:
            raise ValueError(
                f"Expected 2-D or 3-D array, got {data.ndim}-D with shape {data.shape}"
            )

    def _encode_single(self, point_cloud: np.ndarray, *, is_query: bool) -> np.ndarray:
        """Encode a single point cloud.

        Parameters
        ----------
        point_cloud : numpy.ndarray
            Shape ``(num_vectors, dimension)``.
        is_query : bool
            If True, uses SUM; if False, uses AVERAGE.

        Returns
        -------
        numpy.ndarray
            Shape ``(output_dimension,)``.
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (N, {self.dimension}), got {point_cloud.shape}")

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        num_points = point_cloud.shape[0]

        num_partitions = self._num_partitions
        proj_dim = self._proj_dim
        out = np.zeros(self._fde_dim, dtype=np.float32)

        for rep in range(self.num_repetitions):
            rng = np.random.default_rng(self._rep_seeds[rep])

            # SimHash
            sim_matrix = simhash_matrix(rng, self.dimension, self.num_simhash_projections)
            sketches = point_cloud @ sim_matrix

            # Projection
            if self._use_identity:
                projected = point_cloud
            else:
                ams_matrix = ams_projection_matrix(rng, self.dimension, proj_dim)
                projected = point_cloud @ ams_matrix

            # Partition assignment
            partition_counts = np.zeros(num_partitions, dtype=np.int32)
            rep_fde = np.zeros(num_partitions * proj_dim, dtype=np.float32)

            for i in range(num_points):
                pidx = partition_index_gray(sketches[i])
                start = pidx * proj_dim
                rep_fde[start : start + proj_dim] += projected[i]
                partition_counts[pidx] += 1

            # AVERAGE encoding (document)
            if not is_query:
                for pidx in range(num_partitions):
                    start = pidx * proj_dim
                    if partition_counts[pidx] > 0:
                        rep_fde[start : start + proj_dim] /= partition_counts[pidx]
                    elif (
                        self.fill_empty_partitions
                        and num_points > 0
                        and self.num_simhash_projections > 0
                    ):
                        # Fill empty partition with the nearest vector
                        distances = np.array(
                            [distance_to_partition(sketches[j], pidx) for j in range(num_points)]
                        )
                        nearest = np.argmin(distances)
                        rep_fde[start : start + proj_dim] = projected[nearest]

            rep_start = rep * num_partitions * proj_dim
            out[rep_start : rep_start + rep_fde.size] = rep_fde

        # Final projection
        if self.final_projection_dimension is not None and self.final_projection_dimension > 0:
            final_rng = np.random.default_rng(self._final_proj_seed)
            out = count_sketch_vector(final_rng, out, self.final_projection_dimension)

        return out

    def _encode_batch(self, point_clouds: np.ndarray, *, is_query: bool) -> np.ndarray:
        """Encode a batch of point clouds using vectorised operations.

        Parameters
        ----------
        point_clouds : numpy.ndarray
            Shape ``(batch_size, num_vectors, dimension)``.
        is_query : bool
            If True, uses SUM; if False, uses AVERAGE.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch_size, output_dimension)``.
        """
        if point_clouds.ndim != 3 or point_clouds.shape[2] != self.dimension:
            raise ValueError(f"Expected shape (B, N, {self.dimension}), got {point_clouds.shape}")

        point_clouds = np.asarray(point_clouds, dtype=np.float32)
        batch_size, num_points, _ = point_clouds.shape

        num_partitions = self._num_partitions
        proj_dim = self._proj_dim

        out = np.zeros((batch_size, self._fde_dim), dtype=np.float32)

        # Reshape for batch matrix multiplication: (batch_size * num_points, dimension)
        flat_points = point_clouds.reshape(-1, self.dimension)
        doc_indices = np.repeat(np.arange(batch_size), num_points)

        for rep in range(self.num_repetitions):
            rng = np.random.default_rng(self._rep_seeds[rep])

            # SimHash
            sim_matrix = simhash_matrix(rng, self.dimension, self.num_simhash_projections)
            all_sketches = flat_points @ sim_matrix  # (B*N, num_projections)

            # Projection
            if self._use_identity:
                all_projected = flat_points
            else:
                ams_matrix = ams_projection_matrix(rng, self.dimension, proj_dim)
                all_projected = flat_points @ ams_matrix  # (B*N, proj_dim)

            # Vectorised partition indices
            part_indices = partition_indices_gray_batch(all_sketches)  # (B*N,)

            # Aggregate into partitions per document
            rep_fde = np.zeros((batch_size, num_partitions, proj_dim), dtype=np.float32)
            partition_counts = np.zeros((batch_size, num_partitions), dtype=np.int32)

            np.add.at(partition_counts, (doc_indices, part_indices), 1)

            # Scatter-add projected vectors
            doc_part = doc_indices * num_partitions + part_indices
            base = doc_part * proj_dim
            flat_rep_fde = rep_fde.reshape(-1)
            for d in range(proj_dim):
                np.add.at(flat_rep_fde, base + d, all_projected[:, d])
            rep_fde = flat_rep_fde.reshape(batch_size, num_partitions, proj_dim)

            # AVERAGE encoding (document)
            if not is_query:
                counts_3d = partition_counts[:, :, np.newaxis]
                np.divide(rep_fde, counts_3d, out=rep_fde, where=counts_3d > 0)

                if self.fill_empty_partitions and self.num_simhash_projections > 0:
                    empty_docs, empty_parts = np.where(partition_counts == 0)
                    for doc_idx, pidx in zip(empty_docs, empty_parts):
                        doc_start = doc_idx * num_points
                        doc_end = doc_start + num_points
                        doc_sketches = all_sketches[doc_start:doc_end]

                        binary_rep = gray_code_to_binary(int(pidx))
                        target_bits = (
                            binary_rep >> np.arange(self.num_simhash_projections - 1, -1, -1)
                        ) & 1
                        distances = np.sum((doc_sketches > 0).astype(int) != target_bits, axis=1)
                        nearest_local = np.argmin(distances)
                        rep_fde[doc_idx, pidx, :] = all_projected[doc_start + nearest_local]

            rep_start = rep * num_partitions * proj_dim
            out[:, rep_start : rep_start + num_partitions * proj_dim] = rep_fde.reshape(
                batch_size, -1
            )

        # Final projection
        if self.final_projection_dimension is not None and self.final_projection_dimension > 0:
            result = np.zeros((batch_size, self.final_projection_dimension), dtype=np.float32)
            for i in range(batch_size):
                final_rng = np.random.default_rng(self._final_proj_seed)
                result[i] = count_sketch_vector(final_rng, out[i], self.final_projection_dimension)
            out = result

        return out

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Muvera("
            f"num_repetitions={self.num_repetitions}, "
            f"num_simhash_projections={self.num_simhash_projections}, "
            f"dimension={self.dimension}, "
            f"projection_type='{self.projection_type}', "
            f"projection_dimension={self.projection_dimension}, "
            f"fill_empty_partitions={self.fill_empty_partitions}, "
            f"final_projection_dimension={self.final_projection_dimension}, "
            f"seed={self.seed}"
            f")"
        )
