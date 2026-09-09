//! Pre-build restriction of primitive input axes.

use std::collections::HashMap;

use crate::primitive::build::{
    PrimitiveBuildInput, PrimitiveBuildSpec, PrimitiveBuildValue, SweepMode,
};

use super::MacroDesignVariableCondition;

pub(super) type PrimitiveInputConditions = HashMap<String, Vec<MacroDesignVariableCondition>>;

pub(super) fn apply_prebuild_conditions(
    spec: &PrimitiveBuildSpec,
    input: &mut PrimitiveBuildInput,
    conditions: &PrimitiveInputConditions,
) {
    if conditions.is_empty() {
        return;
    }
    match spec.sweep_mode {
        SweepMode::Cartesian => apply_cartesian(input, conditions),
        SweepMode::Aligned => apply_aligned(input, conditions),
    }
}

fn apply_cartesian(input: &mut PrimitiveBuildInput, conditions: &PrimitiveInputConditions) {
    for (name, restrictions) in conditions {
        let Some(value) = input.values.get_mut(name) else {
            continue;
        };
        match value {
            PrimitiveBuildValue::Scalar(scalar) => {
                if !accepts_all(restrictions, *scalar) {
                    *value = PrimitiveBuildValue::Vector(Vec::new());
                }
            }
            PrimitiveBuildValue::Vector(values) => {
                values.retain(|value| accepts_all(restrictions, *value));
            }
        }
    }
}

fn apply_aligned(input: &mut PrimitiveBuildInput, conditions: &PrimitiveInputConditions) {
    let vector_lengths = input
        .values
        .values()
        .filter_map(|value| match value {
            PrimitiveBuildValue::Scalar(_) => None,
            PrimitiveBuildValue::Vector(values) => Some(values.len()),
        })
        .collect::<Vec<_>>();
    let row_count = vector_lengths.iter().copied().max().unwrap_or(1);
    if vector_lengths.iter().any(|length| *length != row_count) {
        return;
    }

    let mut keep = vec![true; row_count];
    for (name, restrictions) in conditions {
        let Some(value) = input.values.get(name) else {
            continue;
        };
        match value {
            PrimitiveBuildValue::Scalar(value) => {
                if !accepts_all(restrictions, *value) {
                    keep.fill(false);
                }
            }
            PrimitiveBuildValue::Vector(values) => {
                for (index, value) in values.iter().enumerate() {
                    keep[index] &= accepts_all(restrictions, *value);
                }
            }
        }
    }

    let has_vectors = !vector_lengths.is_empty();
    for value in input.values.values_mut() {
        if let PrimitiveBuildValue::Vector(values) = value {
            let mut index = 0;
            values.retain(|_| {
                let retain = keep[index];
                index += 1;
                retain
            });
        }
    }
    if !has_vectors && keep.first() == Some(&false) {
        if let Some((name, _)) = conditions.iter().next() {
            if input.values.contains_key(name) {
                input
                    .values
                    .insert(name.clone(), PrimitiveBuildValue::Vector(Vec::new()));
            }
        }
    }
}

fn accepts_all(conditions: &[MacroDesignVariableCondition], value: f64) -> bool {
    conditions.iter().all(|condition| condition.accepts(value))
}

#[cfg(test)]
mod tests {
    use crate::primitive::build::{PrimitiveBuildInputKind, PrimitiveBuildInputSpec, SweepMode};

    use super::*;

    fn spec(sweep_mode: SweepMode) -> PrimitiveBuildSpec {
        PrimitiveBuildSpec {
            inputs: vec![
                PrimitiveBuildInputSpec {
                    name: "x".to_owned(),
                    kind: PrimitiveBuildInputKind::Vector,
                    required: true,
                    source: None,
                },
                PrimitiveBuildInputSpec {
                    name: "y".to_owned(),
                    kind: PrimitiveBuildInputKind::Vector,
                    required: true,
                    source: None,
                },
            ],
            sweep_mode,
            derived: Vec::new(),
            lut: Vec::new(),
            columns: Vec::new(),
        }
    }

    fn input() -> PrimitiveBuildInput {
        PrimitiveBuildInput::new(HashMap::from([
            (
                "x".to_owned(),
                PrimitiveBuildValue::Vector(vec![1.0, 2.0, 3.0]),
            ),
            (
                "y".to_owned(),
                PrimitiveBuildValue::Vector(vec![10.0, 20.0, 30.0]),
            ),
        ]))
    }

    #[test]
    fn cartesian_restricts_only_the_bound_axis() {
        let mut input = input();
        apply_prebuild_conditions(
            &spec(SweepMode::Cartesian),
            &mut input,
            &HashMap::from([(
                "x".to_owned(),
                vec![MacroDesignVariableCondition::range(Some(2.0), Some(3.0))],
            )]),
        );

        assert_eq!(
            input.values["x"],
            PrimitiveBuildValue::Vector(vec![2.0, 3.0])
        );
        assert_eq!(
            input.values["y"],
            PrimitiveBuildValue::Vector(vec![10.0, 20.0, 30.0])
        );
    }

    #[test]
    fn aligned_applies_one_common_mask_to_every_vector() {
        let mut input = input();
        apply_prebuild_conditions(
            &spec(SweepMode::Aligned),
            &mut input,
            &HashMap::from([(
                "x".to_owned(),
                vec![MacroDesignVariableCondition::allowed_values([1.0, 3.0])],
            )]),
        );

        assert_eq!(
            input.values["x"],
            PrimitiveBuildValue::Vector(vec![1.0, 3.0])
        );
        assert_eq!(
            input.values["y"],
            PrimitiveBuildValue::Vector(vec![10.0, 30.0])
        );
    }

    #[test]
    fn empty_intersection_leaves_zero_aligned_rows() {
        let mut input = input();
        apply_prebuild_conditions(
            &spec(SweepMode::Aligned),
            &mut input,
            &HashMap::from([(
                "x".to_owned(),
                vec![
                    MacroDesignVariableCondition::range(Some(2.0), None),
                    MacroDesignVariableCondition::range(None, Some(1.0)),
                ],
            )]),
        );

        assert_eq!(input.values["x"], PrimitiveBuildValue::Vector(Vec::new()));
        assert_eq!(input.values["y"], PrimitiveBuildValue::Vector(Vec::new()));
    }
}
