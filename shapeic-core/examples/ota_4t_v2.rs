use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shapeic_core::analysis::{
    AcMetricSet, AdaptiveAcConfig, AdaptiveAcOutcome, AdaptiveAcPolicy, AnalysisMode,
    AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::compact_model::MosDeviceCapacitances;
use shapeic_core::exploration::candidate::{CandidatePoint, CandidateSet};
use shapeic_core::netlist::names::small_signal_param_name;
use shapeic_core::primitive::build::{
    PrimitiveBuildInput, PrimitiveBuildValue, build_candidate_set_for_primitive,
};
use shapeic_core::testbench::{AcAnalysis, PreparedAcTestbench, TransferFunction};
use shapeic_core::utils::linspace;
use shapeic_lut::{LookupTable, MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_mna::numeric::NumericMnaSystem;

const TAIL_CURRENT: f64 = 20.0e-6;
const VOUT: f64 = 1.0;
const VDD: f64 = 1.5;
const VIN: f64 = 0.9;
const VTAIL_START: f64 = 0.65;
const VTAIL_STOP: f64 = 0.79;
const VTAIL_POINTS: usize = 1000;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
const AC_COARSE_POINTS_PER_DECADE: usize = 4;
const AC_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
const AC_MAX_REFINEMENT_STEPS: usize = 32;
const MNA_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";
const DIFF_PAIR_GM_COLUMN: &str = "gm__xdp__m1";
const DIFF_PAIR_RO_COLUMN: &str = "ro__xdp__m1";
const CURRENT_MIRROR_GM_COLUMN: &str = "gm__xcm__m1";
const CURRENT_MIRROR_RO_COLUMN: &str = "ro__xcm__m1";
const DIFF_PAIR_MOS_CONNECTIONS: [[(&str, &str); 4]; 2] = [
    [("G", "VINP"), ("D", "VOUT"), ("S", "IBIAS"), ("B", "VSS")],
    [("G", "VINN"), ("D", "N1"), ("S", "IBIAS"), ("B", "VSS")],
];
const CURRENT_MIRROR_MOS_CONNECTIONS: [[(&str, &str); 4]; 2] = [
    [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD")],
    [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD")],
];
const ANALYSIS_POLICY: AdaptiveAcPolicy = AdaptiveAcPolicy {
    mode: AnalysisMode::FullInsight,
    targets: AnalysisTargets::NONE,
    metrics: AcMetricSet::ALL,
};

#[derive(Clone, Debug, PartialEq)]
struct PrimitiveCandidate {
    mna_parameters: [f64; 2],
    capacitances: [MosDeviceCapacitances; 2],
}

#[derive(Debug)]
struct OtaAcResult {
    diff_pair_index: usize,
    current_mirror_index: usize,
    outcome: AdaptiveAcOutcome,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let testbench_path = manifest.join("examples/ota_4t/ota_4t.spice");
    let primitives_dir = manifest.join("../analoglib/primitives/");
    let (nmos_path, pmos_path, _physical_path) = lut_paths()?;

    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let lut_load = stage_start.elapsed();
    println!("The LUT load took: {:?}", lut_load);
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;

    let catalog = load_primitive_catalog(&primitives_dir).map_err(|error| format!("{error:?}"))?;
    for primitive in &catalog.list() {
        println!(
            "name: {:?}, description: {:?}",
            primitive.name, primitive.description
        );
    }

    let stage_start = Instant::now();
    let analysis = AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT"),
        adaptive_ac_config(),
        ANALYSIS_POLICY,
    );
    let mut testbench =
        PreparedAcTestbench::from_spice_file(&testbench_path, &MNA_PARAMETER_ORDER, analysis)
            .map_err(|error| {
                io::Error::other(format!("could not prepare OTA AC testbench: {error}"))
            })?;
    let testbench_preparation = stage_start.elapsed();

    let diffpair = catalog
        .get("simplediffpair")
        .ok_or_else(|| "missing simplediffpair primitive".to_string())?;
    let currentmirror = catalog
        .get("simplecurrentmirror")
        .ok_or_else(|| "missing simplecurrentmirror primitive".to_string())?;

    let diffpair_input = PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_string(),
            PrimitiveBuildValue::Scalar(TAIL_CURRENT),
        ),
        ("VINP".to_string(), PrimitiveBuildValue::Scalar(VIN)),
        ("VOUTP".to_string(), PrimitiveBuildValue::Scalar(VOUT)),
        (
            "VTAIL".to_string(),
            PrimitiveBuildValue::Vector(linspace(VTAIL_START, VTAIL_STOP, VTAIL_POINTS)),
        ),
    ]));
    let currentmirror_input = PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_string(),
            PrimitiveBuildValue::Scalar(TAIL_CURRENT),
        ),
        ("VINP".to_string(), PrimitiveBuildValue::Scalar(VOUT)),
        ("VOUTP".to_string(), PrimitiveBuildValue::Scalar(VOUT)),
        ("VDD".to_string(), PrimitiveBuildValue::Scalar(VDD)),
    ]));
    let stage_start = Instant::now();
    let diffpair_candidate_set =
        build_candidate_set_for_primitive(nmos, diffpair, "xdp", diffpair_input)
            .map_err(|error| format!("{error:?}"))?;
    let diffpair_candidate_set_time = stage_start.elapsed();
    println!(
        "Diffpair candidate set took: {:?}",
        diffpair_candidate_set_time
    );

    let stage_start = Instant::now();
    let currentmirror_candidate_set =
        build_candidate_set_for_primitive(pmos, currentmirror, "xcm", currentmirror_input)
            .map_err(|error| format!("{error:?}"))?;
    let currentmirror_candidate_set_time = stage_start.elapsed();
    println!(
        "Current Mirror candidate set took: {:?}",
        currentmirror_candidate_set_time
    );

    let diff_pair_candidates = primitive_candidates(
        &diffpair_candidate_set,
        DIFF_PAIR_INSTANCE,
        DIFF_PAIR_GM_COLUMN,
        DIFF_PAIR_RO_COLUMN,
    )?;
    let current_mirror_candidates = primitive_candidates(
        &currentmirror_candidate_set,
        CURRENT_MIRROR_INSTANCE,
        CURRENT_MIRROR_GM_COLUMN,
        CURRENT_MIRROR_RO_COLUMN,
    )?;
    let mut results = Vec::with_capacity(
        diffpair_candidate_set.points.len() * currentmirror_candidate_set.points.len(),
    );
    let stage_start = Instant::now();
    let mut frequency_evaluations = 0;
    for (diff_pair_index, diff_pair) in diff_pair_candidates.iter().enumerate() {
        for (current_mirror_index, current_mirror) in current_mirror_candidates.iter().enumerate() {
            let parameter_values = [
                diff_pair.mna_parameters[0],
                diff_pair.mna_parameters[1],
                current_mirror.mna_parameters[0],
                current_mirror.mna_parameters[1],
            ];
            let mut candidate_testbench =
                testbench.instantiate(&parameter_values).map_err(|error| {
                io::Error::other(format!(
                    "could not instantiate OTA testbench for diff-pair candidate {diff_pair_index} \
                     and current-mirror candidate {current_mirror_index}: {error}"
                ))
            })?;
            stamp_ota_capacitances(
                candidate_testbench.system_mut(),
                diff_pair,
                current_mirror,
            )
            .map_err(|error| {
                io::Error::other(format!(
                    "could not stamp OTA capacitances for diff-pair candidate {diff_pair_index} \
                     and current-mirror candidate {current_mirror_index}: {error}"
                ))
            })?;
            let outcome = candidate_testbench.analyze().map_err(|error| {
                io::Error::other(format!(
                    "could not analyze OTA testbench for diff-pair candidate {diff_pair_index} \
                     and current-mirror candidate {current_mirror_index}: {error}"
                ))
            })?;
            frequency_evaluations += outcome.frequency_evaluations;
            results.push(OtaAcResult {
                diff_pair_index,
                current_mirror_index,
                outcome,
            });
        }
    }
    let ac_evaluation = stage_start.elapsed();

    print_results(&results);
    println!("Testbench preparation took: {testbench_preparation:?}");
    println!("AC evaluation took: {ac_evaluation:?}");
    println!("Frequency evaluations: {frequency_evaluations}");
    let total = total_start.elapsed();
    println!("Total time: {total:?}");
    Ok(())
}

fn adaptive_ac_config() -> AdaptiveAcConfig {
    AdaptiveAcConfig {
        min_frequency_hz: AC_MIN_HZ,
        max_frequency_hz: AC_MAX_HZ,
        coarse_points_per_decade: AC_COARSE_POINTS_PER_DECADE,
        crossing_relative_tolerance: AC_CROSSING_RELATIVE_TOLERANCE,
        max_refinement_steps: AC_MAX_REFINEMENT_STEPS,
        retain_samples: false,
    }
}

fn primitive_candidates(
    candidates: &CandidateSet,
    instance: &str,
    gm_column: &str,
    ro_column: &str,
) -> Result<Vec<PrimitiveCandidate>, io::Error> {
    candidates
        .points
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let value = |column: &str| candidate_value(candidates, candidate, index, column);
            Ok(PrimitiveCandidate {
                mna_parameters: [value(gm_column)?, value(ro_column)?],
                capacitances: [
                    candidate_capacitances(candidates, candidate, index, instance, "m1")?,
                    candidate_capacitances(candidates, candidate, index, instance, "m2")?,
                ],
            })
        })
        .collect()
}

fn candidate_capacitances(
    candidates: &CandidateSet,
    candidate: &CandidatePoint,
    index: usize,
    instance: &str,
    branch: &str,
) -> Result<MosDeviceCapacitances, io::Error> {
    let intrinsic = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
        .into_iter()
        .map(|parameter| {
            candidate_value(
                candidates,
                candidate,
                index,
                &small_signal_param_name(parameter, instance, branch),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let extrinsic_value = |parameter| {
        candidate_value(
            candidates,
            candidate,
            index,
            &small_signal_param_name(parameter, instance, branch),
        )
    };
    let extrinsic = MosExtrinsicCapacitances {
        cgsol: extrinsic_value("cgsol")?,
        cgdol: extrinsic_value("cgdol")?,
        cjs: extrinsic_value("cjs")?,
        cjd: extrinsic_value("cjd")?,
    };
    MosDeviceCapacitances::from_total_parameters(&intrinsic, extrinsic).map_err(|error| {
        io::Error::other(format!(
            "candidate set '{}' point {index} has invalid MOS capacitances for {instance}.{branch}: {error}",
            candidates.name
        ))
    })
}

fn candidate_value(
    candidates: &CandidateSet,
    candidate: &CandidatePoint,
    index: usize,
    column: &str,
) -> Result<f64, io::Error> {
    candidate
        .values
        .iter()
        .find_map(|(name, value)| (name == column).then_some(*value))
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            io::Error::other(format!(
                "candidate set '{}' point {index} has no finite value for '{column}'",
                candidates.name
            ))
        })
}

fn stamp_ota_capacitances(
    system: &mut NumericMnaSystem,
    diff_pair: &PrimitiveCandidate,
    current_mirror: &PrimitiveCandidate,
) -> Result<(), shapeic_mna::numeric::NumericMnaError> {
    diff_pair.capacitances[0].stamp(system, DIFF_PAIR_MOS_CONNECTIONS[0])?;
    diff_pair.capacitances[1].stamp(system, DIFF_PAIR_MOS_CONNECTIONS[1])?;
    current_mirror.capacitances[0].stamp(system, CURRENT_MIRROR_MOS_CONNECTIONS[0])?;
    current_mirror.capacitances[1].stamp(system, CURRENT_MIRROR_MOS_CONNECTIONS[1])
}

fn print_results(results: &[OtaAcResult]) {
    println!("\nOTA AC results");
    println!(
        "{:<10} {:<10} {:>14} {:>14} {:>14} {:>14}",
        "diff_pair", "mirror", "dc_gain_db", "f3db_hz", "ugf_hz", "pm_deg"
    );
    for result in results {
        let metrics = result.outcome.metrics;
        println!(
            "{:<10} {:<10} {:>14.6} {:>14.6e} {:>14.6e} {:>14.6}",
            result.diff_pair_index,
            result.current_mirror_index,
            metrics.dc_gain_db.unwrap_or(f64::NAN),
            metrics.bandwidth_3db_hz.unwrap_or(f64::NAN),
            metrics.unity_gain_hz.unwrap_or(f64::NAN),
            metrics.phase_margin_deg.unwrap_or(f64::NAN),
        );
    }
}

fn lut_paths() -> Result<(PathBuf, PathBuf, Option<PathBuf>), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments.next().unwrap_or_else(|| OsString::from("ota_4t"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz> [physical.npz]",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    let physical = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into(), physical))
}

#[cfg(test)]
mod tests {
    use shapeic_core::exploration::candidate::{CandidatePoint, CandidateSet};

    use super::{
        ANALYSIS_POLICY, CURRENT_MIRROR_GM_COLUMN, CURRENT_MIRROR_INSTANCE,
        CURRENT_MIRROR_MOS_CONNECTIONS, CURRENT_MIRROR_RO_COLUMN, DIFF_PAIR_GM_COLUMN,
        DIFF_PAIR_INSTANCE, DIFF_PAIR_MOS_CONNECTIONS, DIFF_PAIR_RO_COLUMN, adaptive_ac_config,
        primitive_candidates, small_signal_param_name,
    };
    use shapeic_core::analysis::AcMetric;

    const INTRINSIC: [f64; 9] = [
        10.0e-15, 2.0e-15, 3.0e-15, 1.0e-15, 8.0e-15, 2.0e-15, 1.5e-15, 2.5e-15, 7.0e-15,
    ];

    fn candidate_point(instance: &str, gm_column: &str, ro_column: &str) -> CandidatePoint {
        let mut values = vec![
            (gm_column.to_owned(), 1.2e-3),
            (ro_column.to_owned(), 8.0e4),
        ];
        for branch in ["m1", "m2"] {
            for (parameter, value) in shapeic_lut::MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .zip(INTRINSIC)
            {
                values.push((small_signal_param_name(parameter, instance, branch), value));
            }
            for (parameter, value) in [
                ("cgsol", 1.0e-15),
                ("cgdol", 2.0e-15),
                ("cjs", 3.0e-15),
                ("cjd", 4.0e-15),
            ] {
                values.push((small_signal_param_name(parameter, instance, branch), value));
            }
        }
        CandidatePoint::new(values)
    }

    #[test]
    fn maps_primitive_candidates_to_mna_and_capacitance_values() {
        let candidates = CandidateSet::new(
            "xdp",
            vec![candidate_point(
                DIFF_PAIR_INSTANCE,
                DIFF_PAIR_GM_COLUMN,
                DIFF_PAIR_RO_COLUMN,
            )],
        );
        let mapped = primitive_candidates(
            &candidates,
            DIFF_PAIR_INSTANCE,
            DIFF_PAIR_GM_COLUMN,
            DIFF_PAIR_RO_COLUMN,
        )
        .unwrap();

        assert_eq!(mapped[0].mna_parameters, [1.2e-3, 8.0e4]);
        assert_eq!(mapped[0].capacitances[0], mapped[0].capacitances[1]);
        assert_eq!(mapped[0].capacitances[0].extrinsic.cjd, 4.0e-15);
    }

    #[test]
    fn rejects_missing_or_non_finite_candidate_parameters() {
        let missing = CandidateSet::new(
            "xcm",
            vec![CandidatePoint::new(vec![(
                CURRENT_MIRROR_GM_COLUMN.to_owned(),
                f64::NAN,
            )])],
        );

        let error = primitive_candidates(
            &missing,
            CURRENT_MIRROR_INSTANCE,
            CURRENT_MIRROR_GM_COLUMN,
            CURRENT_MIRROR_RO_COLUMN,
        )
        .unwrap_err();

        assert!(error.to_string().contains(CURRENT_MIRROR_GM_COLUMN));
    }

    #[test]
    fn rejects_missing_branch_capacitances() {
        let mut candidate =
            candidate_point(DIFF_PAIR_INSTANCE, DIFF_PAIR_GM_COLUMN, DIFF_PAIR_RO_COLUMN);
        let missing_column = small_signal_param_name("cjd", DIFF_PAIR_INSTANCE, "m2");
        candidate.values.retain(|(name, _)| name != &missing_column);
        let candidates = CandidateSet::new("xdp", vec![candidate]);

        let error = primitive_candidates(
            &candidates,
            DIFF_PAIR_INSTANCE,
            DIFF_PAIR_GM_COLUMN,
            DIFF_PAIR_RO_COLUMN,
        )
        .unwrap_err();

        assert!(error.to_string().contains(&missing_column));
    }

    #[test]
    fn uses_the_base_ota_terminal_connections_for_all_four_devices() {
        assert_eq!(
            DIFF_PAIR_MOS_CONNECTIONS,
            [
                [("G", "VINP"), ("D", "VOUT"), ("S", "IBIAS"), ("B", "VSS"),],
                [("G", "VINN"), ("D", "N1"), ("S", "IBIAS"), ("B", "VSS"),],
            ]
        );
        assert_eq!(
            CURRENT_MIRROR_MOS_CONNECTIONS,
            [
                [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD"),],
                [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD"),],
            ]
        );
    }

    #[test]
    fn configures_all_ac_metrics_for_the_capacitance_aware_flow() {
        for metric in [
            AcMetric::DcGainDb,
            AcMetric::Bandwidth3DbHz,
            AcMetric::UnityGainHz,
            AcMetric::PhaseMarginDeg,
        ] {
            assert!(ANALYSIS_POLICY.metrics.contains(metric));
        }
        adaptive_ac_config().validate().unwrap();
    }
}
