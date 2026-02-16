use ndarray::ArrayView2;

use crate::gray_code::append_to_gray_code;

/// Compute Gray-code partition index from a single sketch vector.
pub fn partition_index_gray(sketch: &[f32]) -> u32 {
    let mut partition_index: u32 = 0;
    for &val in sketch {
        partition_index = append_to_gray_code(partition_index, val > 0.0);
    }
    partition_index
}

/// Compute Gray-code partition indices for a batch of sketch vectors.
/// `sketches` has shape (N, num_projections).
pub fn partition_indices_gray_batch(sketches: ArrayView2<f32>) -> Vec<u32> {
    let n = sketches.nrows();
    let num_projections = sketches.ncols();
    let mut result = vec![0u32; n];

    for bit_idx in 0..num_projections {
        for i in 0..n {
            let bit = sketches[[i, bit_idx]] > 0.0;
            result[i] = append_to_gray_code(result[i], bit);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_partition_index_single() {
        let sketch = vec![1.0f32, -1.0, 0.5];
        let idx = partition_index_gray(&sketch);
        // bits: true, false, true -> gray code
        assert!(idx < 8);
    }

    #[test]
    fn test_partition_indices_batch() {
        let sketches = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let indices = partition_indices_gray_batch(sketches.view());
        assert_eq!(indices.len(), 3);
        for &idx in &indices {
            assert!(idx < 4);
        }
    }
}
