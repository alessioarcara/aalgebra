mod matrix;
mod vector;
mod linalg;

pub use matrix::Matrix;
pub use vector::Vector;
pub use linalg::{multiply_matrices, gauss_elimination};
