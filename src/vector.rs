use std::ops::{Add, Mul};

#[derive(Debug, PartialEq)]
pub struct Vector<const N: usize>(pub [f64; N]);

#[macro_export]
macro_rules! vector {
    ($($x:expr),*) => {
        Vector([$($x as f64),*])
    };
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
