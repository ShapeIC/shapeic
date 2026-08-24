use ndarray::{Array2, IxDyn};

use crate::LayoutError;
use crate::model::{
    LayoutAwareAdmittance, LayoutAwarePoint, PhysicalPoint, PhysicalPrimitive, PortAdmittance,
};
use crate::validation::{validate_device_correction_matrix, validate_port_matrix};

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

    pub fn query_layout_aware(
        &self,
        point: LayoutAwarePoint,
    ) -> Result<LayoutAwareAdmittance, LayoutError> {
        let correction = self.device_capacitance_correction.as_ref().ok_or_else(|| {
            LayoutError::DeviceCorrectionUnavailable {
                primitive: self.name.clone(),
            }
        })?;
        let interconnect = self.query(point.physical_point())?;
        let brackets = [
            bracket(&self.name, "length", &self.lengths, point.length)?,
            bracket(
                &self.name,
                "finger_width",
                &self.finger_widths,
                point.finger_width,
            )?,
            bracket(
                &self.name,
                "nf",
                &correction.finger_counts,
                f64::from(point.nf),
            )?,
            bracket(&self.name, "vbs", &correction.vbs, point.vbs)?,
            bracket(&self.name, "vgs", &correction.vgs, point.vgs)?,
            bracket(&self.name, "vds", &correction.vds, point.vds)?,
        ];
        let port_count = self.ports.len();
        let mut device_capacitance_correction = Array2::zeros((port_count, port_count));
        for (li, lw) in brackets[0].weighted_indices() {
            for (wi, ww) in brackets[1].weighted_indices() {
                for (ni, nw) in brackets[2].weighted_indices() {
                    for (bi, bw) in brackets[3].weighted_indices() {
                        for (gi, gw) in brackets[4].weighted_indices() {
                            for (di, dw) in brackets[5].weighted_indices() {
                                let weight = lw * ww * nw * bw * gw * dw;
                                for row in 0..port_count {
                                    for column in 0..port_count {
                                        device_capacitance_correction[(row, column)] += weight
                                            * correction.capacitance
                                                [IxDyn(&[li, wi, ni, bi, gi, di, row, column])];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        validate_device_correction_matrix(
            &device_capacitance_correction,
            &self.name,
            "interpolated device capacitance correction",
        )?;
        Ok(LayoutAwareAdmittance {
            ports: interconnect.ports,
            interconnect_conductance: interconnect.conductance,
            interconnect_capacitance: interconnect.capacitance,
            device_capacitance_correction,
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
    use ndarray::{Array5, ArrayD, IxDyn};

    use super::{LayoutAwarePoint, PhysicalPoint, PhysicalPrimitive};
    use crate::LayoutError;
    use crate::model::DeviceCapacitanceCorrection;

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
            device_capacitance_correction: None,
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

    #[test]
    fn interpolates_all_six_layout_aware_axes() {
        let shape = [2, 2, 2, 2, 2, 2, 3, 3];
        let correction = ArrayD::from_shape_fn(IxDyn(&shape), |index| {
            let value = 1.0
                + index[0] as f64
                + 2.0 * index[1] as f64
                + 4.0 * index[2] as f64
                + 8.0 * index[3] as f64
                + 16.0 * index[4] as f64
                + 32.0 * index[5] as f64;
            match (index[6], index[7]) {
                (0, 0) | (1, 1) | (2, 2) => value,
                (0, 1) | (1, 2) | (2, 0) => -value,
                _ => 0.0,
            }
        });
        let primitive = PhysicalPrimitive {
            name: "test".to_owned(),
            ports: vec!["P".to_owned(), "N".to_owned(), "B".to_owned()],
            lengths: vec![1.0, 2.0],
            finger_widths: vec![1.0, 3.0],
            finger_counts: vec![1.0, 2.0, 4.0],
            conductance: Array5::zeros((2, 2, 3, 3, 3)),
            capacitance: Array5::zeros((2, 2, 3, 3, 3)),
            device_capacitance_correction: Some(DeviceCapacitanceCorrection {
                finger_counts: vec![1.0, 4.0],
                vbs: vec![-1.0, 0.0],
                vgs: vec![0.0, 2.0],
                vds: vec![0.0, 4.0],
                capacitance: correction,
            }),
        };

        let result = primitive
            .query_layout_aware(LayoutAwarePoint::new(1.5, 2.0, 2, -0.5, 1.0, 2.0))
            .expect("six-dimensional point should interpolate");
        assert_eq!(result.ports, ["P", "N", "B"]);
        assert!((result.device_capacitance_correction[(0, 0)] - 31.833333333333332).abs() < 1e-12);
        assert_eq!(
            result.device_capacitance_correction[(0, 1)],
            -result.device_capacitance_correction[(0, 0)]
        );
        assert_ne!(
            result.device_capacitance_correction[(0, 1)],
            result.device_capacitance_correction[(1, 0)]
        );
        assert_eq!(
            result.total_capacitance(),
            &result.interconnect_capacitance + &result.device_capacitance_correction
        );

        let error = primitive
            .query_layout_aware(LayoutAwarePoint::new(1.5, 2.0, 2, -0.5, 2.1, 2.0))
            .expect_err("layout-aware bias extrapolation must fail");
        assert!(matches!(
            error,
            LayoutError::OutOfPhysicalRange { axis: "vgs", .. }
        ));
    }

    #[test]
    fn reports_unavailable_device_correction_for_v1_primitive() {
        let error = primitive()
            .query_layout_aware(LayoutAwarePoint::new(1.0, 1.0, 1, 0.0, 0.0, 0.0))
            .expect_err("v1 primitive must not support layout-aware queries");
        assert!(matches!(
            error,
            LayoutError::DeviceCorrectionUnavailable { .. }
        ));
    }
}
