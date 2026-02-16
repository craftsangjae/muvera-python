use ndarray::{ArrayView1, ArrayView2, ArrayViewMut3};

/// Scatter-add projected vectors into partitions.
///
/// `rep_fde` has shape (batch_size, num_partitions, proj_dim).
/// `doc_indices` and `part_indices` have shape (N,).
/// `projected` has shape (N, proj_dim).
pub fn scatter_add_partitions(
    mut rep_fde: ArrayViewMut3<f32>,
    doc_indices: ArrayView1<u32>,
    part_indices: ArrayView1<u32>,
    projected: ArrayView2<f32>,
) {
    let n = doc_indices.len();
    let proj_dim = projected.ncols();

    for i in 0..n {
        let doc = doc_indices[i] as usize;
        let part = part_indices[i] as usize;
        for d in 0..proj_dim {
            rep_fde[[doc, part, d]] += projected[[i, d]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_scatter_add_basic() {
        let mut rep_fde = Array3::<f32>::zeros((2, 4, 3));
        let doc_indices = ndarray::array![0u32, 0, 1];
        let part_indices = ndarray::array![1u32, 1, 2];
        let projected = ndarray::array![
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];

        scatter_add_partitions(
            rep_fde.view_mut(),
            doc_indices.view(),
            part_indices.view(),
            projected.view(),
        );

        assert_eq!(rep_fde[[0, 1, 0]], 5.0); // 1.0 + 4.0
        assert_eq!(rep_fde[[0, 1, 1]], 7.0); // 2.0 + 5.0
        assert_eq!(rep_fde[[1, 2, 0]], 7.0);
    }
}
