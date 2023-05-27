mod linalg;
mod matrix;
mod vector;

pub use linalg::{gauss_elimination, gram_schmidt, multiply_matrices};
pub use matrix::Matrix;
pub use vector::Vector;

#[macro_export]
macro_rules! vector {
    ($($x:expr),*) => {
        $crate::Vector([$($x as f64),*])
    };
}

#[macro_export]
macro_rules! matrix {
    ($($($x:expr),*);*) => {
        $crate::Matrix([
            $([$($x as f64),*]),*
        ])
    };
}
