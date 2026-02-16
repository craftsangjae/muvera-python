"""Type stubs for Rust-accelerated MuVERA kernels."""

import numpy as np
import numpy.typing as npt

def partition_index_gray(sketch: npt.NDArray[np.float32]) -> int: ...
def partition_indices_gray_batch(
    sketches: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint32]: ...
def scatter_add_partitions(
    rep_fde: npt.NDArray[np.float32],
    doc_indices: npt.NDArray[np.uint32],
    part_indices: npt.NDArray[np.uint32],
    projected: npt.NDArray[np.float32],
) -> None: ...
def aggregate_single(
    sketches: npt.NDArray[np.float32],
    projected: npt.NDArray[np.float32],
    num_partitions: int,
    proj_dim: int,
    is_query: bool,
    fill_empty: bool,
    num_simhash_projections: int,
) -> npt.NDArray[np.float32]: ...
def fill_empty_partitions_batch(
    rep_fde: npt.NDArray[np.float32],
    partition_counts: npt.NDArray[np.int32],
    all_sketches: npt.NDArray[np.float32],
    all_projected: npt.NDArray[np.float32],
    doc_boundaries: npt.NDArray[np.int64],
    num_simhash_projections: int,
    num_partitions: int,
    proj_dim: int,
) -> None: ...
