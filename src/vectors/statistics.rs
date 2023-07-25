//! Statistical distances between vectors.

use crate::{number::Float, Number};

/// Helper to compute the mean of a slice of numbers
fn mean<T: Number, U: Float>(xs: &[T]) -> U {
    xs.iter().map(|x| U::from(*x)).sum::<U>() / U::from(xs.len())
}

/// Helper to compute the standard deviation of a slice of numbers
///
/// This computes the population standard deviation, not the sample standard
/// deviation.
fn std_dev<T: Number, U: Float>(xs: &[T]) -> U {
    let xs_mean: U = mean(xs);

    (xs.iter()
        .map(|x| (U::from(*x) - xs_mean).powi(2))
        .sum::<U>()
        / U::from(xs.len()))
    .sqrt()
}

/// Helper to compute the covariance between two slices of numbers
///
/// This computes the population covariance, not the sample covariance.
fn covariance<T: Number, U: Float>(x: &[T], y: &[T]) -> U {
    let x_mean: U = mean(x);
    let y_mean: U = mean(y);

    (x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (U::from(*xi) - x_mean) * (U::from(*yi) - y_mean)))
    .sum::<U>()
        / U::from(x.len())
}

/// Computes the Pearson distance between two vectors.
///
/// The Pearson distance is defined as `1.0 - r` where `r` is the Pearson
/// correlation coefficient.
///
/// The Pearson correlation coefficient is defined as the covariance between the
/// two vectors divided by the product of their standard deviations.
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::pearson;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0, 8.0];
/// let y: Vec<f64> = vec![0.11, 0.12, 0.13, 0.15, 0.18];
///
/// let distance: f64 = pearson(&x, &y);
///
/// assert!((distance - 0.0).abs() < f64::EPSILON);
/// ```
///
/// # References
///
/// * [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
pub fn pearson<T: Number, U: Float>(x: &[T], y: &[T]) -> U {
    let covariance: U = covariance(x, y);

    let std_dev_x: U = std_dev(x);
    let std_dev_y: U = std_dev(y);

    // Compute the Pearson correlation coefficient, `r`
    // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Definition
    let r = covariance / (std_dev_x * std_dev_y);

    // We are defining "Pearson distance" as one minus the Pearson correlation
    // coefficient
    U::one() - r
}
