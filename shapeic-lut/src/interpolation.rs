use crate::{Axis, DeviceLut, Expr, LutError, OperatingPoint};

#[derive(Clone, Copy, Debug)]
struct Bracket {
    lower: usize,
    upper: usize,
    upper_weight: f64,
}

impl DeviceLut {
    /// Interpolate a direct LUT parameter at a physical operating point.
    pub fn query_parameter(
        &self,
        point: &OperatingPoint,
        parameter: &str,
    ) -> Result<f64, LutError> {
        self.query_expression(point, &self.parameter_expression(parameter)?)
    }

    /// Evaluate an expression on LUT corners and linearly interpolate its value in four dimensions.
    pub fn query_expression(
        &self,
        point: &OperatingPoint,
        expression: &Expr,
    ) -> Result<f64, LutError> {
        let [length, vbs, vgs, vds] = Axis::ALL.map(|axis| self.bracket(axis, point.value(axis)));
        let brackets = [length?, vbs?, vgs?, vds?];

        let mut result = 0.0;
        for corner in 0_u8..16 {
            let mut index = [0; 4];
            let mut weight = 1.0;
            for dimension in 0..4 {
                let upper = corner & (1 << dimension) != 0;
                let bracket = brackets[dimension];
                if upper {
                    index[dimension] = bracket.upper;
                    weight *= bracket.upper_weight;
                } else {
                    index[dimension] = bracket.lower;
                    weight *= 1.0 - bracket.upper_weight;
                }
            }

            if weight != 0.0 {
                result += weight * expression.evaluate_at(self, index)?;
            }
        }

        if result.is_finite() {
            Ok(result)
        } else {
            Err(LutError::NonFinite {
                model: self.name.clone(),
                context: format!("interpolated expression '{expression}'"),
            })
        }
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
}
