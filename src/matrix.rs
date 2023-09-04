use crate::linalg::gauss_elimination;
use std::ops::{Index, IndexMut};

pub type SquareMatrix<const N: usize> = Matrix<N, N>;

#[derive(Debug, PartialEq)]
pub struct Matrix<const M: usize, const N: usize>(pub [[f64; N]; M]);

impl<const M: usize, const N: usize> Index<usize> for Matrix<M, N> {
    type Output = [f64; N];

    fn index(&self, index: usize) -> &[f64; N] {
        &self.0[index]
    }
}

impl<const M: usize, const N: usize> IndexMut<usize> for Matrix<M, N> {
    fn index_mut(&mut self, index: usize) -> &mut [f64; N] {
        &mut self.0[index]
    }
}

impl<const M: usize, const N: usize> Clone for Matrix<M, N> {
    fn clone(&self) -> Self {
        let mut result = [[0.0; N]; M];
        (0..M).for_each(|i| (0..N).for_each(|j| result[i][j] = self[i][j]));
        Matrix(result)
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn transpose(&self) -> Matrix<N, M> {
        let mut result = [[0.0; M]; N];
        (0..M).for_each(|i| (0..N).for_each(|j| result[j][i] = self[i][j]));
        Matrix(result)
    }
}

impl<const N: usize> Matrix<N, N> {
    pub fn inverse(&self) -> Option<Self> {
        match gauss_elimination(self, &identity_matrix::<N>()) {
            Ok((_, inverse)) => Some(inverse),
            Err(e) => {
                println!("{e}");
                None
            }
        }
    }

    pub fn determinant(&self) -> f64 {
        fn _determinant(a: &[Vec<f64>]) -> f64 {
            let n = a.len();

            if n == 2 {
                a[0][0] * a[1][1] - a[1][0] * a[0][1]
            } else {
                let mut det = 0.0;
                for i in 0..n {
                    let mut submatrix = vec![vec![0.0; n - 1]; n - 1];
                    for j in 1..n {
                        let mut k = 0;
                        for (l, row) in a.iter().enumerate().take(n) {
                            if l != i {
                                submatrix[k][j - 1] = row[j];
                                k += 1;
                            }
                        }
                    }
                    det += a[i][0] * if i % 2 == 0 { 1.0 } else { -1.0 } * _determinant(&submatrix);
                }
                det
            }
        }
        let a_vec: Vec<Vec<f64>> = self.0.iter().map(|row| row.to_vec()).collect();
        _determinant(&a_vec)
    }
}

pub fn identity_matrix<const N: usize>() -> SquareMatrix<N> {
    let mut id = [[0.0; N]; N];
    (0..N).for_each(|i| id[i][i] = 1.0);
    Matrix(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix, multiply_matrices};

    #[test]
    fn test_matrix_multiplication() {
        let a = matrix!(1, 2; 3, 4);
        let b = matrix!(2, 3; 4, 5);
        assert_eq!(multiply_matrices(&a, &b), matrix!(10, 13; 22, 29));
    }

    #[test]
    fn test_matrix_transpose() {
        let a = matrix!(1, 2; 3, 4; 5, 6);
        assert_eq!(a.transpose(), matrix!(1, 3, 5; 2, 4, 6));
    }

    #[test]
    fn test_matrix_inverse() {
        let a = matrix!(2, 3; 4, 5);
        let b = a.inverse().unwrap();
        let id = identity_matrix::<2>();
        assert_eq!(multiply_matrices(&a, &b), id);
        assert_eq!(multiply_matrices(&b, &a), id);
    }

    #[test]
    fn test_matrix_determinant() {
        let a = matrix!(2, 3, 4; 5, 6, 7; 8, 9, 9);
        assert_eq!(a.determinant(), 3.0);
        assert_eq!(a.determinant(), a.transpose().determinant());
    }
}
