use crate::{
    matrix::{identity_matrix, SquareMatrix},
    vector::zeros_vector,
    Matrix, Vector,
};

pub fn multiply_matrices<const M: usize, const N: usize, const P: usize>(
    a: &Matrix<M, N>,
    b: &Matrix<N, P>,
) -> Matrix<M, P> {
    let mut result = [[0.0; P]; M];
    for i in 0..M {
        for j in 0..P {
            for k in 0..N {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    Matrix(result)
}

fn row_subtract<const N: usize, const P: usize>(
    i: usize,
    j: usize,
    pivot: f64,
    a: &mut SquareMatrix<N>,
    b: &mut Matrix<N, P>,
) {
    let scale = -a[i][j] / pivot;
    for k in 0..N {
        a[i][k] += a[j][k] * scale;
    }
    for k in 0..P {
        b[i][k] += b[j][k] * scale;
    }
}

fn echelon_form<const N: usize, const P: usize>(
    a: &SquareMatrix<N>,
    b: &Matrix<N, P>,
) -> Result<(SquareMatrix<N>, Matrix<N, P>), &'static str> {
    let mut a = a.clone();
    let mut b = b.clone();

    for j in 0..N {
        // row swap
        let mut max_index = j;
        let mut max_value = a[j][j].abs();

        for i in (j + 1)..N {
            let abs_value = a[i][j].abs();
            if abs_value > max_value {
                max_index = i;
                max_value = abs_value;
                break;
            }
        }

        if max_value == 0.0 {
            return Err("Singular matrix");
        }
        if max_index != j {
            a.0.swap(j, max_index);
            b.0.swap(j, max_index);
        }

        // row subtract
        for i in (j + 1)..N {
            row_subtract(i, j, max_value, &mut a, &mut b);
        }
    }
    Ok((a, b))
}

pub fn gauss_elimination<const N: usize, const P: usize>(
    a: &SquareMatrix<N>,
    b: &Matrix<N, P>,
) -> Result<(SquareMatrix<N>, Matrix<N, P>), &'static str> {
    let (mut a, mut b) = echelon_form(a, b)?;

    // normalize
    for j in 0..N {
        let pivot = a[j][j];
        a[j].iter_mut().for_each(|x| *x /= pivot);
        b[j].iter_mut().for_each(|x| *x /= pivot);
    }

    // back substitution
    for j in (1..N).rev() {
        for i in (0..j).rev() {
            row_subtract(i, j, a[j][j], &mut a, &mut b);
        }
    }
    Ok((a, b))
}

pub fn dot_product<const N: usize>(v1: &Vector<N>, v2: &Vector<N>) -> f64 {
    v1.0.iter().zip(v2.0.iter()).map(|(a, b)| a * b).sum()
}

pub fn gram_schmidt<const N: usize>(v: &[Vector<N>]) -> Vec<Vector<N>> {
    let mut u = Vec::with_capacity(v.len());
    let mut iter = v.iter();

    if let Some(v1) = iter.next() {
        u.push(v1.clone());

        for v in iter {
            let mut ortho = v.clone();
            for a in &u {
                let scale = dot_product(a, v) / dot_product(a, a);
                ortho = ortho + a.clone() * -scale;
            }
            u.push(ortho);
        }
    }
    u
}

pub fn forward_substitution<const N: usize>(l: &SquareMatrix<N>, b: &Vector<N>) -> Vector<N> {
    let mut y = zeros_vector::<N>();
    let mut b = b.clone();

    y[0] = b[0] / l[0][0];
    for k in 0..N - 1 {
        for i in k + 1..N {
            b[i] = b[i] - l[i][k] * y[k];
        }
        y[k + 1] = b[k + 1] / l[k + 1][k + 1];
    }
    y
}

pub fn backward_substitution<const N: usize>(u: &SquareMatrix<N>, y: &Vector<N>) -> Vector<N> {
    let mut x = zeros_vector::<N>();
    let mut y = y.clone();

    for k in (0..N).rev() {
        x[k] = y[k] / u[k][k];
        for i in 0..k {
            y[i] = y[i] - u[i][k] * x[k];
        }
    }
    x
}

pub fn lu_decomposition<const N: usize>(a: &SquareMatrix<N>) -> (SquareMatrix<N>, SquareMatrix<N>) {
    let mut u = a.clone();
    let mut l = identity_matrix();

    for k in 0..N - 1 {
        for i in k + 1..N {
            l[i][k] = u[i][k] / u[k][k];
            for j in 0..=k {
                u[i][j] = 0.;
            }
            for j in k + 1..N {
                u[i][j] -= l[i][k] * u[k][j];
            }
        }
    }
    (l, u)
}

#[cfg(test)]
mod tests {
    use crate::{matrix, vector};

    use super::*;

    #[test]
    fn test_forward_substitution() {
        let l = matrix!(1, 0, 0; 2, 1, 0; 3, 4, 1);
        let b = vector!(1, 3, 7);

        let y = forward_substitution(&l, &b);
        let expected_y = vector!(1, 1, 0);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_backward_substitution() {
        let u = matrix!(1, 2, 3; 0, 1, 4; 0, 0, 1);
        let y = vector!(1, 3, 1);

        let x = backward_substitution(&u, &y);
        let expected_x = vector!(0, -1, 1);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = matrix!(2, 1, 0; 4, 5, 2; 6, 15, 12);
        let (l, u) = lu_decomposition(&a);
        assert_eq!(a, multiply_matrices(&l, &u));
    }
}
