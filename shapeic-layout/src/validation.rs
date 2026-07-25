use ndarray::{Array2, Array5};

use crate::LayoutError;

pub(crate) fn validate_axis(values: &[f64], context: &str, name: &str) -> Result<(), LayoutError> {
    if values.is_empty()
        || values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(LayoutError::schema(
            context,
            format!("axis '{name}' must be finite, non-empty, and strictly increasing"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_grid(
    grid: &Array5<f64>,
    context: &str,
    name: &str,
) -> Result<(), LayoutError> {
    for length in 0..grid.shape()[0] {
        for width in 0..grid.shape()[1] {
            for nf in 0..grid.shape()[2] {
                let matrix = grid
                    .slice(ndarray::s![length, width, nf, .., ..])
                    .to_owned();
                validate_port_matrix(&matrix, context, name)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_port_matrix(
    matrix: &Array2<f64>,
    context: &str,
    name: &str,
) -> Result<(), LayoutError> {
    if matrix.nrows() != matrix.ncols() || matrix.iter().any(|value| !value.is_finite()) {
        return Err(LayoutError::schema(
            context,
            format!("{name} matrix must be square and finite"),
        ));
    }
    let scale = matrix
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
        .max(1.0e-30);
    let tolerance = scale * 1.0e-7;
    for row in 0..matrix.nrows() {
        if matrix[(row, row)] < -tolerance || matrix.row(row).sum().abs() > tolerance {
            return Err(LayoutError::schema(
                context,
                format!("{name} matrix is not a passive nodal matrix"),
            ));
        }
        for column in 0..matrix.ncols() {
            if (matrix[(row, column)] - matrix[(column, row)]).abs() > tolerance
                || (row != column && matrix[(row, column)] > tolerance)
            {
                return Err(LayoutError::schema(
                    context,
                    format!("{name} matrix is not symmetric and passive"),
                ));
            }
        }
    }
    Ok(())
}
