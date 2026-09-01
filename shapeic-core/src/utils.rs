pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (count - 1) as f64;
            (0..count)
                .map(|index| {
                    if index + 1 == count {
                        stop
                    } else {
                        start + step * index as f64
                    }
                })
                .collect()
        }
    }
}

/// Returns `count` values whose base-10 exponents are evenly spaced.
///
/// `start_exponent` and `stop_exponent` are included when at least two values
/// are requested. A count of zero returns an empty vector, while a count of one
/// returns only `10^start_exponent`.
pub fn logspace(start_exponent: f64, stop_exponent: f64, count: usize) -> Vec<f64> {
    linspace(start_exponent, stop_exponent, count)
        .into_iter()
        .map(|exponent| 10.0_f64.powf(exponent))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::logspace;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= 1.0e-12 * scale,
            "expected {expected:e}, got {actual:e}"
        );
    }

    #[test]
    fn logspace_builds_base_ten_values_from_evenly_spaced_exponents() {
        let values = logspace(-5.0, -2.0, 4);
        let expected = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2];

        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn logspace_keeps_a_constant_ratio_for_ten_points() {
        let values = logspace(-5.0, -2.0, 10);
        let expected_ratio = 10.0_f64.powf(1.0 / 3.0);

        assert_close(values[0], 1.0e-5);
        assert_close(values[9], 1.0e-2);
        for pair in values.windows(2) {
            assert_close(pair[1] / pair[0], expected_ratio);
        }
    }

    #[test]
    fn logspace_handles_empty_single_and_descending_ranges() {
        assert!(logspace(-5.0, -2.0, 0).is_empty());
        assert_eq!(logspace(-5.0, -2.0, 1), vec![1.0e-5]);

        let descending = logspace(-2.0, -5.0, 4);
        let expected = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5];
        for (actual, expected) in descending.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }
}
