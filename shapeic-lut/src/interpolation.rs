use crate::{
    Axis, DeviceLut, Expr, LutError, LutPoint, MosCapacitanceMatrix, MosExtrinsicCapacitances,
    OperatingPoint,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bracket {
    pub(crate) lower: usize,
    pub(crate) upper: usize,
    pub(crate) upper_weight: f64,
}

impl Bracket {
    pub(crate) const fn exact(index: usize) -> Self {
        Self {
            lower: index,
            upper: index,
            upper_weight: 0.0,
        }
    }
}

impl DeviceLut {
    /// Interpolate the independent intrinsic coefficients and reconstruct the full matrix.
    pub fn query_capacitance_matrix(
        &self,
        point: &OperatingPoint,
    ) -> Result<MosCapacitanceMatrix, LutError> {
        if self.finger_widths().is_some() {
            return Err(LutError::FingerWidthRequired {
                model: self.name().to_owned(),
            });
        }
        let brackets = self.operating_point_brackets(point)?;
        self.interpolate_capacitance_matrix(&brackets)
    }

    /// Interpolate the independent intrinsic coefficients and reconstruct the full 5-D matrix.
    pub fn query_capacitance_matrix_at(
        &self,
        point: &LutPoint,
    ) -> Result<MosCapacitanceMatrix, LutError> {
        let widths = self
            .finger_widths()
            .ok_or_else(|| LutError::NoFingerWidthAxis {
                model: self.name().to_owned(),
            })?;
        let operating = self.operating_point_brackets(&point.operating_point)?;
        let brackets = [
            operating[0],
            operating[1],
            operating[2],
            operating[3],
            self.bracket_finger_width(widths, point.finger_width)?,
        ];
        self.interpolate_capacitance_matrix(&brackets)
    }

    /// Interpolate the four nf anchors and evaluate total extrinsic capacitances.
    pub fn query_extrinsic_capacitances_at(
        &self,
        point: &LutPoint,
        nf: u32,
    ) -> Result<MosExtrinsicCapacitances, LutError> {
        let widths = self
            .finger_widths()
            .ok_or_else(|| LutError::NoFingerWidthAxis {
                model: self.name().to_owned(),
            })?;
        let operating = self.operating_point_brackets(&point.operating_point)?;
        let brackets = [
            operating[0],
            operating[1],
            operating[2],
            operating[3],
            self.bracket_finger_width(widths, point.finger_width)?,
        ];
        let expressions = self
            .extrinsic_capacitance_sample_expressions()?
            .ok_or_else(|| LutError::ExtrinsicCapacitanceSamplesUnavailable {
                model: self.name().to_owned(),
            })?;
        let references = expressions.iter().collect::<Vec<_>>();
        let values = self.interpolate_many(&brackets, &references)?;
        MosExtrinsicCapacitances::from_nf_samples(nf, &values)
    }

    /// Interpolate a stored or reconstructed parameter at a physical operating point.
    pub fn query_parameter(
        &self,
        point: &OperatingPoint,
        parameter: &str,
    ) -> Result<f64, LutError> {
        if self.finger_widths().is_some() {
            return Err(LutError::FingerWidthRequired {
                model: self.name().to_owned(),
            });
        }
        self.query_expression(point, &self.parameter_expression(parameter)?)
    }

    /// Evaluate an expression on LUT corners and linearly interpolate its value in four dimensions.
    pub fn query_expression(
        &self,
        point: &OperatingPoint,
        expression: &Expr,
    ) -> Result<f64, LutError> {
        if self.finger_widths().is_some() {
            return Err(LutError::FingerWidthRequired {
                model: self.name().to_owned(),
            });
        }
        let brackets = self.operating_point_brackets(point)?;
        self.interpolate(&brackets, expression)
    }

    /// Interpolate a stored or reconstructed parameter from a five-dimensional Shapeic LUT.
    pub fn query_parameter_at(&self, point: &LutPoint, parameter: &str) -> Result<f64, LutError> {
        self.query_expression_at(point, &self.parameter_expression(parameter)?)
    }

    /// Evaluate an expression on the 32 corners surrounding a five-dimensional point.
    pub fn query_expression_at(
        &self,
        point: &LutPoint,
        expression: &Expr,
    ) -> Result<f64, LutError> {
        let widths = self
            .finger_widths()
            .ok_or_else(|| LutError::NoFingerWidthAxis {
                model: self.name().to_owned(),
            })?;
        let [length, vbs, vgs, vds] = self.operating_point_brackets(&point.operating_point)?;
        let brackets = [
            length,
            vbs,
            vgs,
            vds,
            self.bracket_finger_width(widths, point.finger_width)?,
        ];
        self.interpolate(&brackets, expression)
    }

    /// Evaluate several expressions for several points, returning point-major rows.
    pub fn query_many(
        &self,
        points: &[OperatingPoint],
        expressions: &[Expr],
    ) -> Result<Vec<Vec<f64>>, LutError> {
        points
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                expressions
                    .iter()
                    .enumerate()
                    .map(|(expression_index, expression)| {
                        self.query_expression(point, expression)
                            .map_err(|source| LutError::Batch {
                                point_index,
                                expression_index,
                                source: Box::new(source),
                            })
                    })
                    .collect()
            })
            .collect()
    }

    /// Evaluate several expressions for several five-dimensional points.
    pub fn query_many_at(
        &self,
        points: &[LutPoint],
        expressions: &[Expr],
    ) -> Result<Vec<Vec<f64>>, LutError> {
        points
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                expressions
                    .iter()
                    .enumerate()
                    .map(|(expression_index, expression)| {
                        self.query_expression_at(point, expression)
                            .map_err(|source| LutError::Batch {
                                point_index,
                                expression_index,
                                source: Box::new(source),
                            })
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn interpolate(
        &self,
        brackets: &[Bracket],
        expression: &Expr,
    ) -> Result<f64, LutError> {
        self.interpolate_many(brackets, &[expression])
            .map(|mut values| values.remove(0))
    }

    pub(crate) fn interpolate_many(
        &self,
        brackets: &[Bracket],
        expressions: &[&Expr],
    ) -> Result<Vec<f64>, LutError> {
        let mut results = vec![0.0; expressions.len()];
        for corner in 0..(1_usize << brackets.len()) {
            let mut index = vec![0; brackets.len()];
            let mut weight = 1.0;
            for (dimension, bracket) in brackets.iter().copied().enumerate() {
                if corner & (1 << dimension) != 0 {
                    index[dimension] = bracket.upper;
                    weight *= bracket.upper_weight;
                } else {
                    index[dimension] = bracket.lower;
                    weight *= 1.0 - bracket.upper_weight;
                }
            }
            if weight != 0.0 {
                for (result, expression) in results.iter_mut().zip(expressions) {
                    *result += weight * expression.evaluate_at(self, &index)?;
                }
            }
        }

        for (result, expression) in results.iter().zip(expressions) {
            if !result.is_finite() {
                return Err(LutError::NonFinite {
                    model: self.name.clone(),
                    context: format!("interpolated expression '{expression}'"),
                });
            }
        }
        Ok(results)
    }

    fn interpolate_capacitance_matrix(
        &self,
        brackets: &[Bracket],
    ) -> Result<MosCapacitanceMatrix, LutError> {
        let expressions = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
            .map(|name| self.parameter_expression(name))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let references = expressions.iter().collect::<Vec<_>>();
        let values = self.interpolate_many(brackets, &references)?;
        MosCapacitanceMatrix::from_flat(&values)
    }

    pub(crate) fn operating_point_brackets(
        &self,
        point: &OperatingPoint,
    ) -> Result<[Bracket; 4], LutError> {
        let [length, vbs, vgs, vds] = Axis::ALL.map(|axis| self.bracket(axis, point.value(axis)));
        Ok([length?, vbs?, vgs?, vds?])
    }

    fn bracket(&self, axis: Axis, value: f64) -> Result<Bracket, LutError> {
        if !value.is_finite() {
            return Err(LutError::NonFinite {
                model: self.name.clone(),
                context: format!("input coordinate '{axis}'"),
            });
        }

        let values = self.axis(axis);
        let ascending = values.len() == 1 || values[0] < values[values.len() - 1];
        let minimum = values[0].min(values[values.len() - 1]);
        let maximum = values[0].max(values[values.len() - 1]);
        if value < minimum || value > maximum {
            return Err(LutError::OutOfRange {
                model: self.name.clone(),
                axis,
                value,
                minimum,
                maximum,
            });
        }

        if values.len() == 1 {
            return Ok(Bracket {
                lower: 0,
                upper: 0,
                upper_weight: 0.0,
            });
        }

        let upper = if ascending {
            values.partition_point(|candidate| *candidate < value)
        } else {
            values.partition_point(|candidate| *candidate > value)
        };

        if upper < values.len() && values[upper] == value {
            return Ok(Bracket {
                lower: upper,
                upper,
                upper_weight: 0.0,
            });
        }

        let lower = upper.saturating_sub(1);
        let upper = upper.min(values.len() - 1);
        let upper_weight = (value - values[lower]) / (values[upper] - values[lower]);
        Ok(Bracket {
            lower,
            upper,
            upper_weight,
        })
    }

    pub(crate) fn bracket_finger_width(
        &self,
        values: &[f64],
        value: f64,
    ) -> Result<Bracket, LutError> {
        if !value.is_finite() {
            return Err(LutError::NonFinite {
                model: self.name.clone(),
                context: "input coordinate 'finger_width'".to_owned(),
            });
        }
        let minimum = values[0].min(values[values.len() - 1]);
        let maximum = values[0].max(values[values.len() - 1]);
        if value < minimum || value > maximum {
            return Err(LutError::FingerWidthOutOfRange {
                model: self.name.clone(),
                value,
                minimum,
                maximum,
            });
        }
        bracket_monotonic(values, value)
    }
}

fn bracket_monotonic(values: &[f64], value: f64) -> Result<Bracket, LutError> {
    if values.len() == 1 {
        return Ok(Bracket {
            lower: 0,
            upper: 0,
            upper_weight: 0.0,
        });
    }
    let ascending = values[0] < values[values.len() - 1];
    let upper = if ascending {
        values.partition_point(|candidate| *candidate < value)
    } else {
        values.partition_point(|candidate| *candidate > value)
    };
    if upper < values.len() && values[upper] == value {
        return Ok(Bracket {
            lower: upper,
            upper,
            upper_weight: 0.0,
        });
    }
    let lower = upper.saturating_sub(1);
    let upper = upper.min(values.len() - 1);
    Ok(Bracket {
        lower,
        upper,
        upper_weight: (value - values[lower]) / (values[upper] - values[lower]),
    })
}
