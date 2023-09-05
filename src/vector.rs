use std::ops::{Add, Index, IndexMut, Mul};

#[derive(Debug, PartialEq)]
pub struct Vector<const N: usize>(pub [f64; N]);

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        (0..N).for_each(|i| self.0[i] += other.0[i]);
        self
    }
}

impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;

    fn mul(mut self, other: f64) -> Self {
        (0..N).for_each(|i| self.0[i] *= other);
        self
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.0[index]
    }
}

impl<const N: usize> Clone for Vector<N> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

#[allow(dead_code)]
pub fn zeros_vector<const N: usize>() -> Vector<N> {
    Vector([0.; N])
}

#[cfg(test)]
mod tests {
    use crate::{
        linalg::{dot_product, gram_schmidt},
        vector,
    };

    #[test]
    fn test_vector_addition() {
        let v1 = vector!(1, 2, 3);
        let v2 = vector!(4, 5, 6);
        assert_eq!(v1 + v2, vector!(5, 7, 9));
    }

    #[test]
    fn test_vector_multiplication() {
        let v1 = vector!(1, 2, 3);
        let scalar = 3.0;
        assert_eq!(v1 * scalar, vector!(3, 6, 9));
    }

    #[test]
    fn test_vector_clone() {
        let v1 = vector!(1, 2, 3);
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_dot_product() {
        let v1 = vector!(1, 1, 0, 0);
        let v2 = vector!(3, 0, 0, 1);
        assert_eq!(3.0, dot_product(&v1, &v2));
    }

    #[test]
    fn test_gram_schmidt() {
        let v = [vector!(1, 1, 0), vector!(2, 0, 1)];
        let u = [vector!(1, 1, 0), vector!(1, -1, 1)];
        assert_eq!(u, gram_schmidt(&v).as_ref());
    }
}
