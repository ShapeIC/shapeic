use crate::interpolation::Bracket;
use crate::{DeviceLut, Expr, LutError, LutPoint, MosExtrinsicCapacitances, OperatingPoint};

/// Result of sizing a width-dependent LUT device for a requested drain current.
#[derive(Clone, Debug, PartialEq)]
pub struct CurrentSizingResult {
    /// Selected five-dimensional LUT point. `finger_width` is the width of one finger.
    pub point: LutPoint,
    /// Integer number of fingers.
    pub nf: u32,
    /// Complete device width, equal to `point.finger_width * nf`.
    pub total_width: f64,
    /// Drain current requested by the caller.
    pub requested_current: f64,
    /// LUT drain current for one finger at the selected point.
    pub finger_current: f64,
    /// Total LUT drain current, equal to `finger_current * nf`.
    pub predicted_current: f64,
    /// Signed error, equal to `predicted_current - requested_current`.
    pub current_error: f64,
    /// Per-finger values corresponding positionally to the requested expressions.
    pub values: Vec<f64>,
    /// Total extrinsic capacitances evaluated for `nf`, when sampled anchors are available.
    pub extrinsic_capacitances: Option<MosExtrinsicCapacitances>,
}

#[derive(Clone, Copy, Debug)]
struct CurrentCandidate {
    nf: u32,
    finger_width: f64,
}

impl DeviceLut {
    /// Select the smallest integer finger count that can reproduce a requested current.
    ///
    /// The returned expression values are evaluated for one finger. When no integer finger
    /// count can reproduce the current inside the LUT width range, the closest neighboring
    /// boundary is returned. This method only accesses LUT data already held in memory.
    pub fn size_for_current(
        &self,
        operating_point: &OperatingPoint,
        requested_current: f64,
        expressions: &[Expr],
    ) -> Result<CurrentSizingResult, LutError> {
        if !requested_current.is_finite() || requested_current <= 0.0 {
            return Err(LutError::InvalidCurrentTarget {
                model: self.name().to_owned(),
                value: requested_current,
            });
        }

        let widths = self
            .finger_widths()
            .ok_or_else(|| LutError::NoFingerWidthAxis {
                model: self.name().to_owned(),
            })?;
        let operating_brackets = self.operating_point_brackets(operating_point)?;
        let id_expression = self.parameter_expression("id")?;
        let mut curve = widths
            .iter()
            .copied()
            .enumerate()
            .map(|(width_index, width)| {
                let brackets = with_width_bracket(operating_brackets, Bracket::exact(width_index));
                self.interpolate(&brackets, &id_expression)
                    .map(|current| (width, current))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if curve.len() > 1 && curve[0].0 > curve[curve.len() - 1].0 {
            curve.reverse();
        }
        validate_current_curve(self, &curve)?;

        let candidate = select_candidate(self, &curve, requested_current)?;
        let point = LutPoint::new(*operating_point, candidate.finger_width);
        let width_bracket = self.bracket_finger_width(widths, candidate.finger_width)?;
        let final_brackets = with_width_bracket(operating_brackets, width_bracket);
        let extrinsic_expressions = self.extrinsic_capacitance_sample_expressions()?;
        let extrinsic_count = extrinsic_expressions.as_ref().map_or(0, Vec::len);
        let mut requested_expressions = Vec::with_capacity(expressions.len() + extrinsic_count + 1);
        requested_expressions.push(&id_expression);
        requested_expressions.extend(expressions.iter());
        if let Some(sampled) = &extrinsic_expressions {
            requested_expressions.extend(sampled);
        }
        let mut interpolated = self.interpolate_many(&final_brackets, &requested_expressions)?;
        let finger_current = interpolated.remove(0);
        let nf = f64::from(candidate.nf);
        let predicted_current = finger_current * nf;
        let sampled_values = interpolated.split_off(expressions.len());
        let extrinsic_capacitances = extrinsic_expressions
            .map(|_| MosExtrinsicCapacitances::from_nf_samples(candidate.nf, &sampled_values))
            .transpose()?;

        Ok(CurrentSizingResult {
            point,
            nf: candidate.nf,
            total_width: candidate.finger_width * nf,
            requested_current,
            finger_current,
            predicted_current,
            current_error: predicted_current - requested_current,
            values: interpolated,
            extrinsic_capacitances,
        })
    }
}

fn with_width_bracket(operating: [Bracket; 4], width: Bracket) -> [Bracket; 5] {
    [
        operating[0],
        operating[1],
        operating[2],
        operating[3],
        width,
    ]
}

fn validate_current_curve(model: &DeviceLut, curve: &[(f64, f64)]) -> Result<(), LutError> {
    for &(finger_width, current) in curve {
        if current <= 0.0 {
            return Err(LutError::NonPositiveFingerCurrent {
                model: model.name().to_owned(),
                finger_width,
                current,
            });
        }
    }
    for window in curve.windows(2) {
        let [(lower_width, lower_current), (upper_width, upper_current)] = window else {
            unreachable!("windows(2) always contains two elements")
        };
        if upper_current <= lower_current {
            return Err(LutError::NonMonotonicFingerCurrent {
                model: model.name().to_owned(),
                lower_width: *lower_width,
                lower_current: *lower_current,
                upper_width: *upper_width,
                upper_current: *upper_current,
            });
        }
    }
    Ok(())
}

fn select_candidate(
    model: &DeviceLut,
    curve: &[(f64, f64)],
    requested_current: f64,
) -> Result<CurrentCandidate, LutError> {
    let &(minimum_width, minimum_current) = curve
        .first()
        .expect("a loaded LUT axis is always non-empty");
    let &(maximum_width, maximum_current) =
        curve.last().expect("a loaded LUT axis is always non-empty");
    let minimum_exact_nf = checked_ceil_nf(model, requested_current, maximum_current)?;
    let maximum_exact_nf = (requested_current / minimum_current).floor();

    if f64::from(minimum_exact_nf) <= maximum_exact_nf {
        let target_finger_current = requested_current / f64::from(minimum_exact_nf);
        return Ok(CurrentCandidate {
            nf: minimum_exact_nf,
            finger_width: invert_current_curve(curve, target_finger_current),
        });
    }

    let lower_nf = (requested_current / maximum_current).floor();
    let upper_nf = checked_ceil_nf(model, requested_current, minimum_current)?;
    let upper = CurrentCandidate {
        nf: upper_nf,
        finger_width: minimum_width,
    };
    if lower_nf < 1.0 {
        return Ok(upper);
    }

    let lower = CurrentCandidate {
        nf: lower_nf as u32,
        finger_width: maximum_width,
    };
    let lower_error = (maximum_current * f64::from(lower.nf) - requested_current).abs();
    let upper_error = (minimum_current * f64::from(upper.nf) - requested_current).abs();
    if lower_error <= upper_error {
        Ok(lower)
    } else {
        Ok(upper)
    }
}

fn checked_ceil_nf(
    model: &DeviceLut,
    requested_current: f64,
    finger_current: f64,
) -> Result<u32, LutError> {
    let nf = (requested_current / finger_current).ceil().max(1.0);
    if nf > f64::from(u32::MAX) {
        return Err(LutError::FingerCountOverflow {
            model: model.name().to_owned(),
            requested_current,
            maximum_fingers: u32::MAX,
        });
    }
    Ok(nf as u32)
}

fn invert_current_curve(curve: &[(f64, f64)], target: f64) -> f64 {
    let upper = curve.partition_point(|(_, current)| *current < target);
    if upper == 0 {
        return curve[0].0;
    }
    if upper == curve.len() {
        return curve[curve.len() - 1].0;
    }
    if curve[upper].1 == target {
        return curve[upper].0;
    }

    let (lower_width, lower_current) = curve[upper - 1];
    let (upper_width, upper_current) = curve[upper];
    let upper_weight = (target - lower_current) / (upper_current - lower_current);
    lower_width + upper_weight * (upper_width - lower_width)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::LutArray;

    fn model(widths: Option<Vec<f64>>, currents: Vec<f64>) -> DeviceLut {
        let width_count = currents.len();
        let shape = IxDyn(&[1, 1, 1, 1, width_count]);
        let gm = currents.iter().map(|current| current * 10.0).collect();
        let parameters = BTreeMap::from([
            (
                "gm".to_owned(),
                LutArray::F64(ArrayD::from_shape_vec(shape.clone(), gm).expect("gm shape")),
            ),
            (
                "id".to_owned(),
                LutArray::F64(ArrayD::from_shape_vec(shape, currents).expect("current shape")),
            ),
        ]);
        DeviceLut {
            name: "sizing_fixture".to_owned(),
            axes: std::array::from_fn(|_| vec![0.0]),
            finger_widths: widths,
            parameters,
            parameter_names: vec!["gm".to_owned(), "id".to_owned()],
            device_parameters: BTreeMap::new(),
        }
    }

    fn point() -> OperatingPoint {
        OperatingPoint::new(0.0, 0.0, 0.0, 0.0)
    }

    #[test]
    fn interpolates_width_and_returns_per_finger_values() {
        let model = model(Some(vec![1.0, 2.0, 3.0]), vec![10.0, 20.0, 30.0]);
        let result = model
            .size_for_current(&point(), 25.0, &[Expr::parameter("gm")])
            .expect("exact sizing");

        assert_eq!(result.nf, 1);
        assert_eq!(result.point.finger_width, 2.5);
        assert_eq!(result.total_width, 2.5);
        assert_eq!(result.finger_current, 25.0);
        assert_eq!(result.predicted_current, 25.0);
        assert_eq!(result.current_error, 0.0);
        assert_eq!(result.values, [250.0]);
    }

    #[test]
    fn chooses_the_smallest_exact_finger_count() {
        let model = model(Some(vec![1.0, 2.0, 3.0]), vec![10.0, 20.0, 30.0]);
        let result = model
            .size_for_current(&point(), 40.0, &[])
            .expect("multiple exact finger counts");

        assert_eq!(result.nf, 2);
        assert_eq!(result.point.finger_width, 2.0);
        assert_eq!(result.predicted_current, 40.0);
    }

    #[test]
    fn supports_descending_finger_width_axes() {
        let model = model(Some(vec![3.0, 2.0, 1.0]), vec![30.0, 20.0, 10.0]);
        let result = model
            .size_for_current(&point(), 25.0, &[Expr::parameter("gm")])
            .expect("descending width sizing");

        assert_eq!(result.nf, 1);
        assert_eq!(result.point.finger_width, 2.5);
        assert_eq!(result.predicted_current, 25.0);
        assert_eq!(result.values, [250.0]);
    }

    #[test]
    fn chooses_the_closest_boundary_and_smallest_nf_on_a_tie() {
        let model = model(Some(vec![1.0, 2.0]), vec![9.0, 10.0]);
        let tied = model
            .size_for_current(&point(), 14.0, &[])
            .expect("closest lower boundary");
        assert_eq!(tied.nf, 1);
        assert_eq!(tied.point.finger_width, 2.0);
        assert_eq!(tied.predicted_current, 10.0);
        assert_eq!(tied.current_error, -4.0);

        let closer_upper = model
            .size_for_current(&point(), 15.0, &[])
            .expect("closest upper boundary");
        assert_eq!(closer_upper.nf, 2);
        assert_eq!(closer_upper.point.finger_width, 1.0);
        assert_eq!(closer_upper.predicted_current, 18.0);
        assert_eq!(closer_upper.current_error, 3.0);
    }

    #[test]
    fn returns_the_minimum_single_finger_point_below_the_lut_range() {
        let model = model(Some(vec![1.0, 2.0]), vec![9.0, 10.0]);
        let result = model
            .size_for_current(&point(), 5.0, &[])
            .expect("nearest minimum current");

        assert_eq!(result.nf, 1);
        assert_eq!(result.point.finger_width, 1.0);
        assert_eq!(result.predicted_current, 9.0);
        assert_eq!(result.current_error, 4.0);
    }

    #[test]
    fn rejects_invalid_curves_and_requests() {
        let non_monotonic = model(Some(vec![1.0, 2.0, 3.0]), vec![10.0, 9.0, 20.0]);
        assert!(matches!(
            non_monotonic.size_for_current(&point(), 15.0, &[]),
            Err(LutError::NonMonotonicFingerCurrent { .. })
        ));

        let non_positive = model(Some(vec![1.0, 2.0]), vec![0.0, 10.0]);
        assert!(matches!(
            non_positive.size_for_current(&point(), 5.0, &[]),
            Err(LutError::NonPositiveFingerCurrent { .. })
        ));

        let valid = model(Some(vec![1.0, 2.0]), vec![10.0, 20.0]);
        for requested in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                valid.size_for_current(&point(), requested, &[]),
                Err(LutError::InvalidCurrentTarget { .. })
            ));
        }

        let four_dimensional = model(None, vec![10.0]);
        assert!(matches!(
            four_dimensional.size_for_current(&point(), 10.0, &[]),
            Err(LutError::NoFingerWidthAxis { .. })
        ));
    }

    #[test]
    fn rejects_finger_counts_larger_than_u32() {
        let model = model(Some(vec![1.0, 2.0]), vec![10.0, 20.0]);
        let requested = (f64::from(u32::MAX) + 1.0) * 20.0;
        assert!(matches!(
            model.size_for_current(&point(), requested, &[]),
            Err(LutError::FingerCountOverflow { .. })
        ));
    }
}
