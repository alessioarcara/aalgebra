use crate::{Matrix, Vector};

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
    a: &mut Matrix<N, N>,
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
    a: &Matrix<N, N>,
    b: &Matrix<N, P>,
) -> Result<(Matrix<N, N>, Matrix<N, P>), &'static str> {
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
    a: &Matrix<N, N>,
    b: &Matrix<N, P>,
) -> Result<(Matrix<N, N>, Matrix<N, P>), &'static str> {
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
