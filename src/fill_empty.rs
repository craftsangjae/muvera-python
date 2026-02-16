use ndarray::{ArrayView1, ArrayView2, ArrayViewMut3};

use crate::gray_code::gray_code_to_binary;
use crate::partition::partition_index_gray;

/// Compute Hamming distance between a sketch vector's sign bits and a partition.
fn hamming_distance(sketch: &[f32], partition_index: u32, num_projections: usize) -> u32 {
    let binary_rep = gray_code_to_binary(partition_index);
    let mut dist = 0u32;
    for bit_idx in 0..num_projections {
        let sketch_bit = if sketch[bit_idx] > 0.0 { 1u32 } else { 0u32 };
        let shift = (num_projections - 1 - bit_idx) as u32;
        let target_bit = (binary_rep >> shift) & 1;
        if sketch_bit != target_bit {
            dist += 1;
        }
    }
    dist
}

/// Aggregate vectors into partitions for a single point cloud, applying
/// AVERAGE for documents and filling empty partitions.
///
/// Returns a flat vector of length num_partitions * proj_dim.
pub fn aggregate_single(
    sketches: ArrayView2<f32>,
    projected: ArrayView2<f32>,
    num_partitions: usize,
    proj_dim: usize,
    is_query: bool,
    fill_empty: bool,
    num_simhash_projections: usize,
) -> Vec<f32> {
    let num_points = sketches.nrows();
    let total_dim = num_partitions * proj_dim;
    let mut rep_fde = vec![0.0f32; total_dim];
    let mut partition_counts = vec![0i32; num_partitions];

    // Assign vectors to partitions and accumulate
    for i in 0..num_points {
        let sketch_slice = sketches.row(i);
        let pidx = partition_index_gray(sketch_slice.as_slice().unwrap()) as usize;
        let start = pidx * proj_dim;
        for d in 0..proj_dim {
            rep_fde[start + d] += projected[[i, d]];
        }
        partition_counts[pidx] += 1;
    }

    // For documents: apply average and fill empty partitions
    if !is_query {
        for pidx in 0..num_partitions {
            let start = pidx * proj_dim;
            if partition_counts[pidx] > 0 {
                let count = partition_counts[pidx] as f32;
                for d in 0..proj_dim {
                    rep_fde[start + d] /= count;
                }
            } else if fill_empty && num_points > 0 && num_simhash_projections > 0 {
                // Find nearest vector by Hamming distance
                let mut min_dist = u32::MAX;
                let mut nearest = 0usize;
                for j in 0..num_points {
                    let sketch_slice = sketches.row(j);
                    let dist = hamming_distance(
                        sketch_slice.as_slice().unwrap(),
                        pidx as u32,
                        num_simhash_projections,
                    );
                    if dist < min_dist {
                        min_dist = dist;
                        nearest = j;
                    }
                }
                for d in 0..proj_dim {
                    rep_fde[start + d] = projected[[nearest, d]];
                }
            }
        }
    }

    rep_fde
}

/// Fill empty partitions for a batch of documents.
///
/// `rep_fde` has shape (batch_size, num_partitions, proj_dim).
/// `partition_counts` has shape (batch_size, num_partitions).
/// `all_sketches` has shape (total_vectors, num_simhash_projections).
/// `all_projected` has shape (total_vectors, proj_dim).
/// `doc_boundaries` has shape (batch_size + 1,) with cumulative offsets.
pub fn fill_empty_partitions_batch(
    mut rep_fde: ArrayViewMut3<f32>,
    partition_counts: ArrayView2<i32>,
    all_sketches: ArrayView2<f32>,
    all_projected: ArrayView2<f32>,
    doc_boundaries: ArrayView1<i64>,
    num_simhash_projections: usize,
    num_partitions: usize,
    proj_dim: usize,
) {
    if num_simhash_projections == 0 {
        return;
    }

    let batch_size = partition_counts.nrows();

    for doc_idx in 0..batch_size {
        let doc_start = doc_boundaries[doc_idx] as usize;
        let doc_end = doc_boundaries[doc_idx + 1] as usize;
        if doc_start == doc_end {
            continue;
        }

        for pidx in 0..num_partitions {
            if partition_counts[[doc_idx, pidx]] != 0 {
                continue;
            }

            // Find nearest vector by Hamming distance
            let binary_rep = gray_code_to_binary(pidx as u32);
            let mut min_dist = u32::MAX;
            let mut nearest = doc_start;

            for j in doc_start..doc_end {
                let sketch_slice = all_sketches.row(j);
                let mut dist = 0u32;
                for bit_idx in 0..num_simhash_projections {
                    let sketch_bit = if sketch_slice[bit_idx] > 0.0 { 1u32 } else { 0u32 };
                    let shift = (num_simhash_projections - 1 - bit_idx) as u32;
                    let target_bit = (binary_rep >> shift) & 1;
                    if sketch_bit != target_bit {
                        dist += 1;
                    }
                }
                if dist < min_dist {
                    min_dist = dist;
                    nearest = j;
                }
            }

            for d in 0..proj_dim {
                rep_fde[[doc_idx, pidx, d]] = all_projected[[nearest, d]];
            }
        }
    }
}
