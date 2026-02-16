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
    ams_projection_matrix_from_seed,
    count_sketch_vector_from_seed,
    distance_to_partition,
    gray_code_to_binary,
    partition_index_gray,
    partition_indices_gray_batch,
    simhash_matrix_from_seed,
)


class Muvera:
    """Encoder that converts multi-vector embeddings into Fixed Dimensional Encodings.

    Uses the MuVERA algorithm to encode variable-length multi-vector
    representations (point clouds) into fixed-dimensional single vectors.
    The dot product between encoded vectors approximates the Chamfer
    similarity (MaxSim) between the original multi-vector sets.

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
    >>> # Single document/query
    >>> doc = np.random.randn(80, 128).astype(np.float32)
    >>> query = np.random.randn(32, 128).astype(np.float32)
    >>> doc_fde = encoder.encode_documents(doc)
    >>> query_fde = encoder.encode_queries(query)
    >>> score = query_fde @ doc_fde  # similarity score
    >>>
    >>> # Batch of documents with variable lengths
    >>> docs = [np.random.randn(80, 128).astype(np.float32) for _ in range(5)]
    >>> doc_fdes = encoder.encode_documents(docs)  # (5, output_dimension)
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

    @property
    def output_dimension(self) -> int:
        """Dimension of the encoded output vector."""
        if self.final_projection_dimension is not None and self.final_projection_dimension > 0:
            return self.final_projection_dimension
        return self._fde_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_documents(self, documents: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Encode document point clouds into Fixed Dimensional Encodings.

        Uses AVERAGE aggregation within partitions. Empty partitions are filled
        with the nearest vector when ``fill_empty_partitions=True``.

        Parameters
        ----------
        documents : np.ndarray or list[np.ndarray]
            - Single document: shape ``(num_vectors, dimension)``
            - Batch of documents: ``list[np.ndarray]`` where each element has
              shape ``(num_vectors_i, dimension)``

        Returns
        -------
        np.ndarray
            Encoded FDE vector(s). Single document returns shape ``(output_dimension,)``,
            batch returns shape ``(num_documents, output_dimension)``.
        """
        if isinstance(documents, list):
            return self._encode_batch(documents, is_query=False)
        return self._encode_single(documents, is_query=False)

    def encode_queries(self, queries: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Encode query point clouds into Fixed Dimensional Encodings.

        Uses SUM aggregation within partitions. Empty partitions are not filled.

        Parameters
        ----------
        queries : np.ndarray or list[np.ndarray]
            - Single query: shape ``(num_vectors, dimension)``
            - Batch of queries: ``list[np.ndarray]`` where each element has
              shape ``(num_vectors_i, dimension)``

        Returns
        -------
        np.ndarray
            Encoded FDE vector(s). Single query returns shape ``(output_dimension,)``,
            batch returns shape ``(num_queries, output_dimension)``.
        """
        if isinstance(queries, list):
            return self._encode_batch(queries, is_query=True)
        return self._encode_single(queries, is_query=True)

    # ------------------------------------------------------------------
    # Core Encoding
    # ------------------------------------------------------------------

    def _encode_single(self, point_cloud: np.ndarray, *, is_query: bool) -> np.ndarray:
        """Encode a single point cloud."""
        if point_cloud.ndim != 2 or point_cloud.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (N, {self.dimension}), got {point_cloud.shape}")

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        out = np.zeros(self._fde_dim, dtype=np.float32)

        for rep in range(self.num_repetitions):
            current_seed = self.seed + rep
            sketches = self._compute_sketches(point_cloud, current_seed)
            projected = self._compute_projection(point_cloud, current_seed)
            rep_fde = self._aggregate_single(sketches, projected, is_query)

            rep_start = rep * self._num_partitions * self._proj_dim
            out[rep_start : rep_start + rep_fde.size] = rep_fde

        return self._apply_final_projection(out)

    def _encode_batch(self, point_clouds: list[np.ndarray], *, is_query: bool) -> np.ndarray:
        """Encode a batch of variable-length point clouds."""
        batch_size = len(point_clouds)
        if batch_size == 0:
            return np.zeros((0, self.output_dimension), dtype=np.float32)

        # Validate and flatten
        for i, pc in enumerate(point_clouds):
            if pc.ndim != 2 or pc.shape[1] != self.dimension:
                raise ValueError(f"Element {i}: expected (N, {self.dimension}), got {pc.shape}")

        flat_points, doc_indices, doc_boundaries = self._flatten_batch(point_clouds)
        out = np.zeros((batch_size, self._fde_dim), dtype=np.float32)

        for rep in range(self.num_repetitions):
            current_seed = self.seed + rep
            sketches = self._compute_sketches(flat_points, current_seed)
            projected = self._compute_projection(flat_points, current_seed)
            rep_fde = self._aggregate_batch(
                sketches, projected, doc_indices, doc_boundaries, is_query
            )

            rep_start = rep * self._num_partitions * self._proj_dim
            out[:, rep_start : rep_start + rep_fde.size // batch_size] = rep_fde

        return self._apply_final_projection(out)

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _compute_sketches(self, vectors: np.ndarray, seed: int) -> np.ndarray:
        """Compute SimHash sketches for partition assignment."""
        sim_matrix = simhash_matrix_from_seed(self.dimension, self.num_simhash_projections, seed)
        return vectors @ sim_matrix

    def _compute_projection(self, vectors: np.ndarray, seed: int) -> np.ndarray:
        """Project vectors using identity or AMS sketch."""
        if self._use_identity:
            return vectors

        ams_matrix = ams_projection_matrix_from_seed(self.dimension, self._proj_dim, seed)
        return vectors @ ams_matrix

    def _aggregate_single(
        self, sketches: np.ndarray, projected: np.ndarray, is_query: bool
    ) -> np.ndarray:
        """Aggregate vectors into partitions for a single point cloud."""
        from muvera import _RUST_AVAILABLE

        if _RUST_AVAILABLE:
            from muvera._rust_kernels import aggregate_single as _rs_fn

            return np.asarray(
                _rs_fn(
                    np.ascontiguousarray(sketches, dtype=np.float32),
                    np.ascontiguousarray(projected, dtype=np.float32),
                    self._num_partitions,
                    self._proj_dim,
                    is_query,
                    self.fill_empty_partitions,
                    self.num_simhash_projections,
                ),
                dtype=np.float32,
            )

        return self._aggregate_single_python(sketches, projected, is_query)

    def _aggregate_single_python(
        self, sketches: np.ndarray, projected: np.ndarray, is_query: bool
    ) -> np.ndarray:
        """Aggregate vectors into partitions for a single point cloud (Python fallback)."""
        num_points = sketches.shape[0]
        partition_counts = np.zeros(self._num_partitions, dtype=np.int32)
        rep_fde = np.zeros(self._num_partitions * self._proj_dim, dtype=np.float32)

        # Assign vectors to partitions
        for i in range(num_points):
            pidx = partition_index_gray(sketches[i])
            start = pidx * self._proj_dim
            rep_fde[start : start + self._proj_dim] += projected[i]
            partition_counts[pidx] += 1

        # Apply AVERAGE for documents, SUM for queries
        if not is_query:
            self._apply_average_and_fill(rep_fde, partition_counts, sketches, projected)

        return rep_fde

    def _aggregate_batch(
        self,
        all_sketches: np.ndarray,
        all_projected: np.ndarray,
        doc_indices: np.ndarray,
        doc_boundaries: np.ndarray,
        is_query: bool,
    ) -> np.ndarray:
        """Aggregate vectors into partitions for a batch of point clouds."""
        batch_size = len(doc_boundaries) - 1
        part_indices = partition_indices_gray_batch(all_sketches)
        partition_counts = np.zeros((batch_size, self._num_partitions), dtype=np.int32)
        rep_fde = np.zeros((batch_size, self._num_partitions, self._proj_dim), dtype=np.float32)

        # Count partitions
        np.add.at(partition_counts, (doc_indices, part_indices), 1)

        # Scatter-add projected vectors
        self._scatter_add(rep_fde, doc_indices, part_indices, all_projected)

        # Apply AVERAGE for documents
        if not is_query:
            self._apply_average_batch(rep_fde, partition_counts)
            self._fill_empty_batch(
                rep_fde, partition_counts, all_sketches, all_projected, doc_boundaries
            )

        return rep_fde.reshape(batch_size, -1)

    def _apply_average_and_fill(
        self,
        rep_fde: np.ndarray,
        partition_counts: np.ndarray,
        sketches: np.ndarray,
        projected: np.ndarray,
    ) -> None:
        """Apply AVERAGE aggregation and fill empty partitions for single cloud."""
        num_points = sketches.shape[0]

        for pidx in range(self._num_partitions):
            start = pidx * self._proj_dim
            if partition_counts[pidx] > 0:
                rep_fde[start : start + self._proj_dim] /= partition_counts[pidx]
            elif self.fill_empty_partitions and num_points > 0 and self.num_simhash_projections > 0:
                distances = np.array(
                    [distance_to_partition(sketches[j], pidx) for j in range(num_points)]
                )
                nearest = np.argmin(distances)
                rep_fde[start : start + self._proj_dim] = projected[nearest]

    def _apply_average_batch(self, rep_fde: np.ndarray, partition_counts: np.ndarray) -> None:
        """Apply AVERAGE aggregation for batch."""
        counts_3d = partition_counts[:, :, np.newaxis]
        np.divide(rep_fde, counts_3d, out=rep_fde, where=counts_3d > 0)

    def _fill_empty_batch(
        self,
        rep_fde: np.ndarray,
        partition_counts: np.ndarray,
        all_sketches: np.ndarray,
        all_projected: np.ndarray,
        doc_boundaries: np.ndarray,
    ) -> None:
        """Fill empty partitions for batch."""
        if not self.fill_empty_partitions or self.num_simhash_projections == 0:
            return

        from muvera import _RUST_AVAILABLE

        if _RUST_AVAILABLE:
            from muvera._rust_kernels import fill_empty_partitions_batch as _rs_fn

            _rs_fn(
                rep_fde,
                np.ascontiguousarray(partition_counts, dtype=np.int32),
                np.ascontiguousarray(all_sketches, dtype=np.float32),
                np.ascontiguousarray(all_projected, dtype=np.float32),
                np.ascontiguousarray(doc_boundaries, dtype=np.int64),
                self.num_simhash_projections,
                self._num_partitions,
                self._proj_dim,
            )
            return

        empty_docs, empty_parts = np.where(partition_counts == 0)
        for doc_idx, pidx in zip(empty_docs, empty_parts):
            doc_start, doc_end = doc_boundaries[doc_idx], doc_boundaries[doc_idx + 1]
            if doc_start == doc_end:
                continue

            doc_sketches = all_sketches[doc_start:doc_end]
            binary_rep = gray_code_to_binary(int(pidx))
            target_bits = (binary_rep >> np.arange(self.num_simhash_projections - 1, -1, -1)) & 1
            distances = np.sum((doc_sketches > 0).astype(int) != target_bits, axis=1)
            nearest_local = np.argmin(distances)
            rep_fde[doc_idx, pidx, :] = all_projected[doc_start + nearest_local]

    def _scatter_add(
        self,
        rep_fde: np.ndarray,
        doc_indices: np.ndarray,
        part_indices: np.ndarray,
        all_projected: np.ndarray,
    ) -> None:
        """Scatter-add projected vectors into partitions."""
        from muvera import _RUST_AVAILABLE

        if _RUST_AVAILABLE:
            from muvera._rust_kernels import scatter_add_partitions as _rs_fn

            _rs_fn(
                rep_fde,
                np.ascontiguousarray(doc_indices, dtype=np.uint32),
                np.ascontiguousarray(part_indices, dtype=np.uint32),
                np.ascontiguousarray(all_projected, dtype=np.float32),
            )
            return

        doc_part = doc_indices * self._num_partitions + part_indices
        base = doc_part * self._proj_dim
        flat_rep_fde = rep_fde.reshape(-1)

        for d in range(self._proj_dim):
            np.add.at(flat_rep_fde, base + d, all_projected[:, d])

    def _flatten_batch(
        self, point_clouds: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten batch of point clouds into single array with metadata."""
        doc_lengths = np.array([pc.shape[0] for pc in point_clouds], dtype=np.int32)
        doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
        doc_indices = np.repeat(np.arange(len(point_clouds)), doc_lengths)
        flat_points = np.vstack(point_clouds).astype(np.float32)

        return flat_points, doc_indices, doc_boundaries

    def _apply_final_projection(self, fdes: np.ndarray) -> np.ndarray:
        """Apply optional Count Sketch final projection."""
        if self.final_projection_dimension is None or self.final_projection_dimension <= 0:
            return fdes

        if fdes.ndim == 1:
            return count_sketch_vector_from_seed(fdes, self.final_projection_dimension, self.seed)

        # Batch
        result = np.zeros((fdes.shape[0], self.final_projection_dimension), dtype=np.float32)
        for i in range(fdes.shape[0]):
            result[i] = count_sketch_vector_from_seed(
                fdes[i], self.final_projection_dimension, self.seed
            )
        return result

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
