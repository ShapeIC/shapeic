use ndarray::{Array2, Array5, ArrayD, IxDyn};

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

pub(crate) fn validate_device_correction_grid(
    grid: &ArrayD<f64>,
    context: &str,
    name: &str,
) -> Result<(), LayoutError> {
    if grid.ndim() != 8 {
        return Err(LayoutError::schema(
            context,
            format!("{name} grid must be eight-dimensional"),
        ));
    }
    let shape = grid.shape();
    let port_count = shape[6];
    if port_count == 0 || shape[7] != port_count {
        return Err(LayoutError::schema(
            context,
            format!("{name} matrices must be non-empty and square"),
        ));
    }
    for length in 0..shape[0] {
        for width in 0..shape[1] {
            for nf in 0..shape[2] {
                for vbs in 0..shape[3] {
                    for vgs in 0..shape[4] {
                        for vds in 0..shape[5] {
                            let matrix =
                                Array2::from_shape_fn((port_count, port_count), |(row, column)| {
                                    grid[IxDyn(&[length, width, nf, vbs, vgs, vds, row, column])]
                                });
                            validate_device_correction_matrix(&matrix, context, name)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_device_correction_matrix(
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
    for index in 0..matrix.nrows() {
        if matrix.row(index).sum().abs() > tolerance || matrix.column(index).sum().abs() > tolerance
        {
            return Err(LayoutError::schema(
                context,
                format!("{name} matrix does not conserve charge"),
            ));
        }
    }
    Ok(())
}
