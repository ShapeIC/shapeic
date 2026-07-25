use ndarray::Array2;

use crate::LayoutError;
use crate::model::{PhysicalPoint, PhysicalPrimitive, PortAdmittance};
use crate::validation::validate_port_matrix;

impl PhysicalPrimitive {
    pub fn query(&self, point: PhysicalPoint) -> Result<PortAdmittance, LayoutError> {
        if point.nf == 0 {
            return Err(LayoutError::InvalidFingerCount);
        }
        let brackets = [
            bracket(&self.name, "length", &self.lengths, point.length)?,
            bracket(
                &self.name,
                "finger_width",
                &self.finger_widths,
                point.finger_width,
            )?,
            bracket(&self.name, "nf", &self.finger_counts, f64::from(point.nf))?,
        ];
        let port_count = self.ports.len();
        let mut conductance = Array2::zeros((port_count, port_count));
        let mut capacitance = Array2::zeros((port_count, port_count));

        for (li, lw) in brackets[0].weighted_indices() {
            for (wi, ww) in brackets[1].weighted_indices() {
                for (ni, nw) in brackets[2].weighted_indices() {
                    let weight = lw * ww * nw;
                    for row in 0..port_count {
                        for column in 0..port_count {
                            conductance[(row, column)] +=
                                weight * self.conductance[(li, wi, ni, row, column)];
                            capacitance[(row, column)] +=
                                weight * self.capacitance[(li, wi, ni, row, column)];
                        }
                    }
                }
            }
        }
        validate_port_matrix(&conductance, &self.name, "interpolated conductance")?;
        validate_port_matrix(&capacitance, &self.name, "interpolated capacitance")?;
        Ok(PortAdmittance {
            ports: self.ports.clone(),
            conductance,
            capacitance,
        })
    }
}

#[derive(Clone, Copy)]
struct Bracket {
    lower: usize,
    upper: usize,
    upper_weight: f64,
}

impl Bracket {
    fn weighted_indices(self) -> Vec<(usize, f64)> {
        if self.lower == self.upper {
            vec![(self.lower, 1.0)]
        } else {
            vec![
                (self.lower, 1.0 - self.upper_weight),
                (self.upper, self.upper_weight),
            ]
        }
    }
}

fn bracket(
    primitive: &str,
    axis_name: &'static str,
    axis: &[f64],
    value: f64,
) -> Result<Bracket, LayoutError> {
    let minimum = axis[0];
    let maximum = axis[axis.len() - 1];
    let tolerance = (maximum - minimum)
        .abs()
        .max(minimum.abs())
        .max(maximum.abs())
        * 1.0e-12;
    if !value.is_finite() || value < minimum - tolerance || value > maximum + tolerance {
        return Err(LayoutError::OutOfPhysicalRange {
            primitive: primitive.to_owned(),
            axis: axis_name,
            value,
            minimum,
            maximum,
        });
    }
    let value = value.clamp(minimum, maximum);
    match axis.binary_search_by(|candidate| candidate.total_cmp(&value)) {
        Ok(index) => Ok(Bracket {
            lower: index,
            upper: index,
            upper_weight: 0.0,
        }),
        Err(upper) => {
            let lower = upper - 1;
            Ok(Bracket {
                lower,
                upper,
                upper_weight: (value - axis[lower]) / (axis[upper] - axis[lower]),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array5;

    use super::{PhysicalPoint, PhysicalPrimitive};

    fn primitive() -> PhysicalPrimitive {
        let mut conductance = Array5::zeros((2, 2, 3, 2, 2));
        let mut capacitance = Array5::zeros((2, 2, 3, 2, 2));
        for li in 0..2 {
            for wi in 0..2 {
                for ni in 0..3 {
                    let value = 1.0 + li as f64 + 2.0 * wi as f64 + ni as f64;
                    for matrix in [&mut conductance, &mut capacitance] {
                        matrix[(li, wi, ni, 0, 0)] = value;
                        matrix[(li, wi, ni, 0, 1)] = -value;
                        matrix[(li, wi, ni, 1, 0)] = -value;
                        matrix[(li, wi, ni, 1, 1)] = value;
                    }
                }
            }
        }
        PhysicalPrimitive {
            name: "test".to_owned(),
            ports: vec!["P".to_owned(), "N".to_owned()],
            lengths: vec![1.0, 2.0],
            finger_widths: vec![1.0, 3.0],
            finger_counts: vec![1.0, 2.0, 4.0],
            conductance,
            capacitance,
        }
    }

    #[test]
    fn interpolates_length_width_and_integer_nf() {
        let result = primitive()
            .query(PhysicalPoint::new(1.5, 2.0, 3))
            .expect("point should interpolate");
        assert_eq!(result.ports, ["P", "N"]);
        assert!((result.conductance[(0, 0)] - 4.0).abs() < 1.0e-12);
        assert_eq!(result.conductance[(0, 1)], -result.conductance[(0, 0)]);
    }

    #[test]
    fn rejects_extrapolation() {
        let error = primitive()
            .query(PhysicalPoint::new(1.5, 2.0, 21))
            .expect_err("nf outside the axis must fail");
        assert!(error.to_string().contains("outside"));
    }

    #[test]
    fn accepts_roundoff_at_a_physical_boundary() {
        primitive()
            .query(PhysicalPoint::new(1.0 - 1.0e-13, 1.0, 1))
            .expect("small coordinate roundoff should clamp to the boundary");
    }
}
