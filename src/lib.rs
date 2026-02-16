mod fill_empty;
mod gray_code;
mod partition;
mod scatter;

use numpy::ndarray::{ArrayView1, ArrayView2, ArrayViewMut3};
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute a Gray-code-based partition index from a single sketch vector.
#[pyfunction]
fn partition_index_gray<'py>(sketch: PyReadonlyArray1<'py, f32>) -> u32 {
    let view = sketch.as_array();
    partition::partition_index_gray(view.as_slice().unwrap())
}

/// Compute Gray-code partition indices for a batch of sketch vectors.
#[pyfunction]
fn partition_indices_gray_batch<'py>(
    py: Python<'py>,
    sketches: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray1<u32>> {
    let view: ArrayView2<f32> = sketches.as_array();
    let result = partition::partition_indices_gray_batch(view);
    PyArray1::from_vec(py, result)
}

/// Scatter-add projected vectors into partitions (in-place).
#[pyfunction]
fn scatter_add_partitions<'py>(
    rep_fde: &Bound<'py, PyArray3<f32>>,
    doc_indices: PyReadonlyArray1<'py, u32>,
    part_indices: PyReadonlyArray1<'py, u32>,
    projected: PyReadonlyArray2<'py, f32>,
) {
    let rep_fde_rw: ArrayViewMut3<f32> =
        unsafe { rep_fde.as_array_mut() };
    let doc_view: ArrayView1<u32> = doc_indices.as_array();
    let part_view: ArrayView1<u32> = part_indices.as_array();
    let proj_view: ArrayView2<f32> = projected.as_array();
    scatter::scatter_add_partitions(rep_fde_rw, doc_view, part_view, proj_view);
}

/// Aggregate vectors into partitions for a single point cloud.
#[pyfunction]
fn aggregate_single<'py>(
    py: Python<'py>,
    sketches: PyReadonlyArray2<'py, f32>,
    projected: PyReadonlyArray2<'py, f32>,
    num_partitions: usize,
    proj_dim: usize,
    is_query: bool,
    fill_empty: bool,
    num_simhash_projections: usize,
) -> Bound<'py, PyArray1<f32>> {
    let sk_view: ArrayView2<f32> = sketches.as_array();
    let pr_view: ArrayView2<f32> = projected.as_array();
    let result = fill_empty::aggregate_single(
        sk_view,
        pr_view,
        num_partitions,
        proj_dim,
        is_query,
        fill_empty,
        num_simhash_projections,
    );
    PyArray1::from_vec(py, result)
}

/// Fill empty partitions for a batch of documents (in-place).
#[pyfunction]
fn fill_empty_partitions_batch<'py>(
    rep_fde: &Bound<'py, PyArray3<f32>>,
    partition_counts: PyReadonlyArray2<'py, i32>,
    all_sketches: PyReadonlyArray2<'py, f32>,
    all_projected: PyReadonlyArray2<'py, f32>,
    doc_boundaries: PyReadonlyArray1<'py, i64>,
    num_simhash_projections: usize,
    num_partitions: usize,
    proj_dim: usize,
) {
    let rep_fde_rw: ArrayViewMut3<f32> =
        unsafe { rep_fde.as_array_mut() };
    let counts_view: ArrayView2<i32> = partition_counts.as_array();
    let sk_view: ArrayView2<f32> = all_sketches.as_array();
    let pr_view: ArrayView2<f32> = all_projected.as_array();
    let boundaries_view: ArrayView1<i64> = doc_boundaries.as_array();
    fill_empty::fill_empty_partitions_batch(
        rep_fde_rw,
        counts_view,
        sk_view,
        pr_view,
        boundaries_view,
        num_simhash_projections,
        num_partitions,
        proj_dim,
    );
}

/// Rust-accelerated kernels for MuVERA.
#[pymodule]
fn _rust_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(partition_index_gray, m)?)?;
    m.add_function(wrap_pyfunction!(partition_indices_gray_batch, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_add_partitions, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_single, m)?)?;
    m.add_function(wrap_pyfunction!(fill_empty_partitions_batch, m)?)?;
    Ok(())
}
