use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shapeic_core::analysis::{
    AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcOutcome, AdaptiveAcPolicy, AnalysisMode,
    AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::exploration::candidate::CandidateSet;
use shapeic_core::primitive::build::{
    PrimitiveBuildInput, PrimitiveBuildValue, build_candidate_set_for_primitive,
};
use shapeic_core::testbench::{AcAnalysis, PreparedAcTestbench, TransferFunction};
use shapeic_core::utils::linspace;
use shapeic_lut::LookupTable;

const TAIL_CURRENT: f64 = 20.0e-6;
const VOUT: f64 = 1.0;
const VDD: f64 = 1.5;
const VIN: f64 = 0.9;
const VTAIL_START: f64 = 0.65;
const VTAIL_STOP: f64 = 0.79;
const VTAIL_POINTS: usize = 10;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
const AC_COARSE_POINTS_PER_DECADE: usize = 4;
const AC_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
const AC_MAX_REFINEMENT_STEPS: usize = 32;
const MNA_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];
const DIFF_PAIR_GM_COLUMN: &str = "gm__xdp__m1";
const DIFF_PAIR_RO_COLUMN: &str = "ro__xdp__m1";
const CURRENT_MIRROR_GM_COLUMN: &str = "gm__xcm__m1";
const CURRENT_MIRROR_RO_COLUMN: &str = "ro__xcm__m1";
const ANALYSIS_POLICY: AdaptiveAcPolicy = AdaptiveAcPolicy {
    mode: AnalysisMode::FullInsight,
    targets: AnalysisTargets::NONE,
    metrics: AcMetricSet::from_metric(AcMetric::DcGainDb),
};

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

    let diff_pair_parameters = block_parameter_values(
        &diffpair_candidate_set,
        DIFF_PAIR_GM_COLUMN,
        DIFF_PAIR_RO_COLUMN,
    )?;
    let current_mirror_parameters = block_parameter_values(
        &currentmirror_candidate_set,
        CURRENT_MIRROR_GM_COLUMN,
        CURRENT_MIRROR_RO_COLUMN,
    )?;
    let mut results = Vec::with_capacity(
        diffpair_candidate_set.points.len() * currentmirror_candidate_set.points.len(),
    );
    let stage_start = Instant::now();
    let mut frequency_evaluations = 0;
    for (diff_pair_index, diff_pair) in diff_pair_parameters.iter().enumerate() {
        for (current_mirror_index, current_mirror) in current_mirror_parameters.iter().enumerate() {
            let parameter_values = [
                diff_pair[0],
                diff_pair[1],
                current_mirror[0],
                current_mirror[1],
            ];
            let testbench = testbench.instantiate(&parameter_values).map_err(|error| {
                io::Error::other(format!(
                    "could not instantiate OTA testbench for diff-pair candidate {diff_pair_index} \
                     and current-mirror candidate {current_mirror_index}: {error}"
                ))
            })?;
            let outcome = testbench.analyze().map_err(|error| {
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

fn block_parameter_values(
    candidates: &CandidateSet,
    gm_column: &str,
    ro_column: &str,
) -> Result<Vec<[f64; 2]>, io::Error> {
    candidates
        .points
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let value = |column: &str| {
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
            };
            Ok([value(gm_column)?, value(ro_column)?])
        })
        .collect()
}

fn print_results(results: &[OtaAcResult]) {
    println!("\nOTA AC results");
    println!("{:<10} {:<10} {:>14}", "diff_pair", "mirror", "dc_gain_db");
    for result in results {
        let dc_gain_db = result.outcome.metrics.dc_gain_db.unwrap_or(f64::NAN);
        println!(
            "{:<10} {:<10} {:>14.6}",
            result.diff_pair_index, result.current_mirror_index, dc_gain_db
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
        ANALYSIS_POLICY, CURRENT_MIRROR_GM_COLUMN, CURRENT_MIRROR_RO_COLUMN, DIFF_PAIR_GM_COLUMN,
        DIFF_PAIR_RO_COLUMN, adaptive_ac_config, block_parameter_values,
    };
    use shapeic_core::analysis::AcMetric;

    #[test]
    fn maps_primitive_candidate_columns_to_mna_values() {
        let candidates = CandidateSet::new(
            "xdp",
            vec![CandidatePoint::new(vec![
                (DIFF_PAIR_GM_COLUMN.to_owned(), 1.2e-3),
                (DIFF_PAIR_RO_COLUMN.to_owned(), 8.0e4),
            ])],
        );

        assert_eq!(
            block_parameter_values(&candidates, DIFF_PAIR_GM_COLUMN, DIFF_PAIR_RO_COLUMN).unwrap(),
            vec![[1.2e-3, 8.0e4]]
        );
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

        let error =
            block_parameter_values(&missing, CURRENT_MIRROR_GM_COLUMN, CURRENT_MIRROR_RO_COLUMN)
                .unwrap_err();

        assert!(error.to_string().contains(CURRENT_MIRROR_GM_COLUMN));
    }

    #[test]
    fn configures_the_current_capacitance_free_flow_for_dc_gain_only() {
        assert!(ANALYSIS_POLICY.metrics.contains(AcMetric::DcGainDb));
        assert!(!ANALYSIS_POLICY.metrics.contains(AcMetric::Bandwidth3DbHz));
        adaptive_ac_config().validate().unwrap();
    }
}
