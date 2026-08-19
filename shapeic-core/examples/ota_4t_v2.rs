use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shapeic_core::analysis::{
    AcCompletion, AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcOutcome, AdaptiveAcPolicy,
    AnalysisMode, AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::compact_model::MosDeviceCapacitances;
use shapeic_core::exploration::candidate::{
    CandidateEquality, CandidateJoinError, CandidatePairJoin, CandidatePairSelection,
    CandidatePoint, CandidateSet,
};
use shapeic_core::netlist::names::small_signal_param_name;
use shapeic_core::primitive::build::{
    PrimitiveBuildInput, PrimitiveBuildValue, build_candidate_set_for_primitive,
};
use shapeic_core::testbench::{
    AcAnalysis, PreparedAcTestbench, TransferFunction, TransferPolarity,
};
use shapeic_core::utils::linspace;
use shapeic_lut::{LookupTable, MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_mna::numeric::NumericMnaSystem;

use shapeic_core::exploration::filter::{
      CandidateFilter, retain_candidate_set,
  };

const TAIL_CURRENT: f64 = 20.0e-6;
const VOUT: f64 = 1.0;
const VOUT_START: f64 = 0.9;
const VOUT_STOP: f64 = 1.1;
const VOUT_POINTS: usize = 10;
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
const MIN_DC_GAIN_DB: f64 = 25.0;
const MIN_BANDWIDTH_3DB_HZ: f64 = 1.0e6;
const MIN_UNITY_GAIN_HZ: f64 = 1.0e7;
const MIN_PHASE_MARGIN_DEG: f64 = 45.0;
const MNA_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";
const DIFF_PAIR_GM_COLUMN: &str = "gm__xdp__m1";
const DIFF_PAIR_RO_COLUMN: &str = "ro__xdp__m1";
const CURRENT_MIRROR_GM_COLUMN: &str = "gm__xcm__m1";
const CURRENT_MIRROR_RO_COLUMN: &str = "ro__xcm__m1";
const DIFF_PAIR_VOUT_COLUMN: &str = "xdp.voutp";
const DIFF_PAIR_VBIAS_COLUMN: &str = "xdp.vtail";
const CURRENT_MIRROR_VOUT_COLUMN: &str = "xcm.voutp";
const DIFF_PAIR_WIDTH_COLUMN: &str = "width__xdp__m1";
const DIFF_PAIR_LENGTH_COLUMN: &str = "length__xdp__m1";
const DIFF_PAIR_NF_COLUMN: &str = "nf__xdp__m1";
const CURRENT_MIRROR_WIDTH_COLUMN: &str = "width__xcm__m1";
const CURRENT_MIRROR_LENGTH_COLUMN: &str = "length__xcm__m1";
const CURRENT_MIRROR_NF_COLUMN: &str = "nf__xcm__m1";
const DIFF_PAIR_MOS_CONNECTIONS: [[(&str, &str); 4]; 2] = [
    [("G", "VINP"), ("D", "VOUT"), ("S", "IBIAS"), ("B", "VSS")],
    [("G", "VINN"), ("D", "N1"), ("S", "IBIAS"), ("B", "VSS")],
];
const CURRENT_MIRROR_MOS_CONNECTIONS: [[(&str, &str); 4]; 2] = [
    [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD")],
    [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD")],
];
const MAX_DIFF_PAIR_WIDTH: f64 = 100.0e-6;
const MAX_CURRENT_MIRROR_WIDTH: f64 = 100.0e-6;

#[derive(Clone, Debug, PartialEq)]
struct PrimitiveCandidate {
    mna_parameters: [f64; 2],
    capacitances: [MosDeviceCapacitances; 2],
}

#[derive(Debug)]
struct OtaAcResult {
    selection: CandidatePairSelection,
    outcome: AdaptiveAcOutcome,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct AcRejectionCounts {
    dc_gain: usize,
    bandwidth_3db: usize,
    unity_gain: usize,
    phase_margin: usize,
}

impl AcRejectionCounts {
    fn record(
        &mut self,
        completion: AcCompletion,
        targets_passed: bool,
    ) -> Result<bool, io::Error> {
        match (completion, targets_passed) {
            (AcCompletion::Complete, true) => Ok(true),
            (AcCompletion::PrunedAfter(metric), false) => {
                match metric {
                    AcMetric::DcGainDb => self.dc_gain += 1,
                    AcMetric::Bandwidth3DbHz => self.bandwidth_3db += 1,
                    AcMetric::UnityGainHz => self.unity_gain += 1,
                    AcMetric::PhaseMarginDeg => self.phase_margin += 1,
                }
                Ok(false)
            }
            (AcCompletion::Complete, false) => Err(io::Error::other(
                "AC analysis completed despite failing one or more targets",
            )),
            (AcCompletion::PrunedAfter(metric), true) => Err(io::Error::other(format!(
                "AC analysis pruned after {} despite passing all evaluated targets",
                metric.label()
            ))),
        }
    }

    const fn total(&self) -> usize {
        self.dc_gain + self.bandwidth_3db + self.unity_gain + self.phase_margin
    }
}

fn main() -> Result<(), Box<dyn Error>> {

    //Some paths definitions
    let total_start = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let testbench_path = manifest.join("examples/ota_4t/ota_4t.spice");
    let primitives_dir = manifest.join("../analoglib/primitives/");
    let (nmos_path, pmos_path, _physical_path) = lut_paths()?;

    //Load the LUTs and the models
    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let lut_load = stage_start.elapsed();
    println!("The LUT load took: {:?}", lut_load);
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;

    //Load the primitives catalog
    let catalog = load_primitive_catalog(&primitives_dir).map_err(|error| format!("{error:?}"))?;
    for primitive in &catalog.list() {
        println!(
            "name: {:?}, description: {:?}",
            primitive.name, primitive.description
        );
    }

    //Configure de AC analysis
    let stage_start = Instant::now();
    let adaptive_ac_config = AdaptiveAcConfig {
        min_frequency_hz: AC_MIN_HZ,
        max_frequency_hz: AC_MAX_HZ,
        coarse_points_per_decade: AC_COARSE_POINTS_PER_DECADE,
        crossing_relative_tolerance: AC_CROSSING_RELATIVE_TOLERANCE,
        max_refinement_steps: AC_MAX_REFINEMENT_STEPS,
        retain_samples: false,
    };
    let adaptive_ac_policy = AdaptiveAcPolicy {
    mode: AnalysisMode::Prune,
    targets: AnalysisTargets {
        min_dc_gain_db: Some(MIN_DC_GAIN_DB),
        min_bandwidth_3db_hz: Some(MIN_BANDWIDTH_3DB_HZ),
        min_unity_gain_hz: Some(MIN_UNITY_GAIN_HZ),
        min_phase_margin_deg: Some(MIN_PHASE_MARGIN_DEG),
    },
    metrics: AcMetricSet::ALL,
    };
    let analysis = AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT").with_polarity(TransferPolarity::Negative),
        adaptive_ac_config,
        adaptive_ac_policy,
    );

    //Define the Testbench
    let mut testbench =
        PreparedAcTestbench::from_spice_file(&testbench_path, &MNA_PARAMETER_ORDER, analysis)
            .map_err(|error| {
                io::Error::other(format!("could not prepare OTA AC testbench: {error}"))
            })?;
    let testbench_preparation = stage_start.elapsed();

    //Define the primitives
    let diffpair = catalog
        .get("simplediffpair")
        .ok_or_else(|| "missing simplediffpair primitive".to_string())?;
    let currentmirror = catalog
        .get("simplecurrentmirror")
        .ok_or_else(|| "missing simplecurrentmirror primitive".to_string())?;

    //Define the primtiives input
    let diffpair_input = PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_string(),
            PrimitiveBuildValue::Scalar(TAIL_CURRENT),
        ),
        ("VINP".to_string(), PrimitiveBuildValue::Scalar(VIN)),
        (
            "VOUTP".to_string(),
            PrimitiveBuildValue::Vector(linspace(VOUT_START, VOUT_STOP, VOUT_POINTS)),
        ),
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
        (
            "VOUTP".to_string(),
            PrimitiveBuildValue::Vector(linspace(VOUT_START, VOUT_STOP, VOUT_POINTS)),
        ),
        ("VDD".to_string(), PrimitiveBuildValue::Scalar(VDD)),
    ]));

    //Generate the Candidates
    let stage_start = Instant::now();
    let mut diffpair_candidate_set =
        build_candidate_set_for_primitive(nmos, diffpair, "xdp", diffpair_input)
            .map_err(|error| format!("{error:?}"))?;
    let diffpair_candidate_set_time = stage_start.elapsed();
    let diffpair_filter_report = retain_candidate_set(
        &mut diffpair_candidate_set,
        &[CandidateFilter::at_most(
            DIFF_PAIR_WIDTH_COLUMN,
            MAX_DIFF_PAIR_WIDTH,
        )?],
    )?;
    println!(
        "Diffpair candidate set took: {:?}",
        diffpair_candidate_set_time
    );

    let stage_start = Instant::now();
    let mut currentmirror_candidate_set =
        build_candidate_set_for_primitive(pmos, currentmirror, "xcm", currentmirror_input)
            .map_err(|error| format!("{error:?}"))?;
    let currentmirror_candidate_set_time = stage_start.elapsed();
    let currentmirror_filter_report = retain_candidate_set(
        &mut currentmirror_candidate_set,
        &[CandidateFilter::at_most(
            CURRENT_MIRROR_WIDTH_COLUMN,
            MAX_CURRENT_MIRROR_WIDTH,
        )?],
    )?;
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
    let possible_pairs = diffpair_candidate_set
        .points
        .len()
        .checked_mul(currentmirror_candidate_set.points.len())
        .ok_or_else(|| io::Error::other("OTA candidate pair count overflows usize"))?;
    let candidate_pairs =
        ota_candidate_pairs(&diffpair_candidate_set, &currentmirror_candidate_set)
            .map_err(|error| io::Error::other(format!("could not join OTA candidates: {error}")))?;
    let compatible_pairs = candidate_pairs.len();

    //Evaluation
    let mut results = Vec::with_capacity(compatible_pairs);
    let stage_start = Instant::now();
    let mut frequency_evaluations = 0;
    let mut ac_rejections = AcRejectionCounts::default();
    for selection in candidate_pairs {
        let diff_pair = &diff_pair_candidates[selection.left_index];
        let current_mirror = &current_mirror_candidates[selection.right_index];
        let parameter_values = [
            diff_pair.mna_parameters[0],
            diff_pair.mna_parameters[1],
            current_mirror.mna_parameters[0],
            current_mirror.mna_parameters[1],
        ];
        let mut candidate_testbench =
            testbench.instantiate(&parameter_values).map_err(|error| {
                io::Error::other(format!(
                    "could not instantiate OTA testbench for diff-pair candidate {} and \
                     current-mirror candidate {}: {error}",
                    selection.left_index, selection.right_index,
                ))
            })?;
        stamp_ota_capacitances(candidate_testbench.system_mut(), diff_pair, current_mirror)
            .map_err(|error| {
                io::Error::other(format!(
                    "could not stamp OTA capacitances for diff-pair candidate {} and \
                 current-mirror candidate {}: {error}",
                    selection.left_index, selection.right_index,
                ))
            })?;
        let outcome = candidate_testbench.analyze().map_err(|error| {
            io::Error::other(format!(
                "could not analyze OTA testbench for diff-pair candidate {} and \
                 current-mirror candidate {}: {error}",
                selection.left_index, selection.right_index,
            ))
        })?;
        frequency_evaluations += outcome.frequency_evaluations;
        if ac_rejections.record(outcome.completion, outcome.targets.passed())? {
            results.push(OtaAcResult { selection, outcome });
        }
    }
    let ac_evaluation = stage_start.elapsed();
    debug_assert_eq!(compatible_pairs, results.len() + ac_rejections.total());

    print_results(
        &results,
        &diffpair_candidate_set,
        &currentmirror_candidate_set,
    )?;
    println!("Diff-pair candidates rejected by width: {}",
        diffpair_filter_report.rejected_count()
    );
    println!("Current-mirror candidates rejected by width: {}",
        currentmirror_filter_report.rejected_count()
    );
    println!("Candidate pairs: {possible_pairs}");
    println!("Compatible candidate pairs: {compatible_pairs}");
    println!(
        "Rejected by shared VOUT: {}",
        possible_pairs - compatible_pairs
    );
    println!("Rejected by DC gain: {}", ac_rejections.dc_gain);
    println!(
        "Rejected by 3 dB bandwidth: {}",
        ac_rejections.bandwidth_3db
    );
    println!("Rejected by UGF: {}", ac_rejections.unity_gain);
    println!("Rejected by phase margin: {}", ac_rejections.phase_margin);
    println!("Accepted candidates: {}", results.len());
    println!("Testbench preparation took: {testbench_preparation:?}");
    println!("AC evaluation took: {ac_evaluation:?}");
    println!("Frequency evaluations: {frequency_evaluations}");
    let total = total_start.elapsed();
    println!("Total time: {total:?}");
    Ok(())
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

fn ota_candidate_pairs<'a>(
    diff_pair: &'a CandidateSet,
    current_mirror: &'a CandidateSet,
) -> Result<CandidatePairJoin<'a>, CandidateJoinError> {
    CandidatePairJoin::new(
        diff_pair,
        current_mirror,
        &[CandidateEquality::new(
            DIFF_PAIR_VOUT_COLUMN,
            CURRENT_MIRROR_VOUT_COLUMN,
        )],
    )
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
        .get(column)
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

fn print_results(
    results: &[OtaAcResult],
    diffpair_candidate_set: &CandidateSet,
    currentmirror_candidate_set: &CandidateSet,
) -> Result<(), io::Error> {
    println!("\nOTA AC results");
    println!(
        "{:<8} {:<8} {:>9} {:>9} {:>10} {:>10} {:>6} {:>10} {:>10} {:>6} {:>12} {:>12} {:>12} {:>10}",
        "dp_idx",
        "cm_idx",
        "vout_v",
        "vbias_v",
        "w_dp_um",
        "l_dp_um",
        "nf_dp",
        "w_cm_um",
        "l_cm_um",
        "nf_cm",
        "dc_gain_db",
        "f3db_hz",
        "ugf_hz",
        "pm_deg",
    );
    for result in results {
        let metrics = result.outcome.metrics;
        let diffpair_point = &diffpair_candidate_set.points[result.selection.left_index];
        let currentmirror_point = &currentmirror_candidate_set.points[result.selection.right_index];
        let diff_value = |column| {
            candidate_value(
                diffpair_candidate_set,
                diffpair_point,
                result.selection.left_index,
                column,
            )
        };
        let mirror_value = |column| {
            candidate_value(
                currentmirror_candidate_set,
                currentmirror_point,
                result.selection.right_index,
                column,
            )
        };
        let vout = diff_value(DIFF_PAIR_VOUT_COLUMN)?;
        debug_assert_eq!(vout, mirror_value(CURRENT_MIRROR_VOUT_COLUMN)?);
        println!(
            "{:<8} {:<8} {:>9.4} {:>9.4} {:>10.4} {:>10.4} {:>6.0} {:>10.4} {:>10.4} {:>6.0} {:>12.6} {:>12.6e} {:>12.6e} {:>10.6}",
            result.selection.left_index,
            result.selection.right_index,
            vout,
            diff_value(DIFF_PAIR_VBIAS_COLUMN)?,
            diff_value(DIFF_PAIR_WIDTH_COLUMN)? * 1.0e6,
            diff_value(DIFF_PAIR_LENGTH_COLUMN)? * 1.0e6,
            diff_value(DIFF_PAIR_NF_COLUMN)?,
            mirror_value(CURRENT_MIRROR_WIDTH_COLUMN)? * 1.0e6,
            mirror_value(CURRENT_MIRROR_LENGTH_COLUMN)? * 1.0e6,
            mirror_value(CURRENT_MIRROR_NF_COLUMN)?,
            metrics.dc_gain_db.unwrap_or(f64::NAN),
            metrics.bandwidth_3db_hz.unwrap_or(f64::NAN),
            metrics.unity_gain_hz.unwrap_or(f64::NAN),
            metrics.phase_margin_deg.unwrap_or(f64::NAN),
        );
    }
    Ok(())
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
        ANALYSIS_POLICY, AcRejectionCounts, CURRENT_MIRROR_GM_COLUMN, CURRENT_MIRROR_INSTANCE,
        CURRENT_MIRROR_LENGTH_COLUMN, CURRENT_MIRROR_MOS_CONNECTIONS, CURRENT_MIRROR_RO_COLUMN,
        CURRENT_MIRROR_VOUT_COLUMN, CURRENT_MIRROR_WIDTH_COLUMN, DIFF_PAIR_GM_COLUMN,
        DIFF_PAIR_INSTANCE, DIFF_PAIR_LENGTH_COLUMN, DIFF_PAIR_MOS_CONNECTIONS,
        DIFF_PAIR_RO_COLUMN, DIFF_PAIR_VOUT_COLUMN, DIFF_PAIR_WIDTH_COLUMN, MIN_BANDWIDTH_3DB_HZ,
        MIN_DC_GAIN_DB, MIN_PHASE_MARGIN_DEG, MIN_UNITY_GAIN_HZ, adaptive_ac_config,
        ota_candidate_pairs, ota_transfer_function, primitive_candidates, small_signal_param_name,
    };
    use shapeic_core::analysis::{AcCompletion, AcMetric, AnalysisMode, AnalysisTargets};
    use shapeic_core::testbench::TransferPolarity;

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

    fn join_candidate(
        vout_column: &str,
        width_column: &str,
        length_column: &str,
        vout: f64,
        width: f64,
        length: f64,
    ) -> CandidatePoint {
        CandidatePoint::new(vec![
            (vout_column.to_owned(), vout),
            (width_column.to_owned(), width),
            (length_column.to_owned(), length),
        ])
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
    fn joins_ota_candidates_by_vout_and_preserves_metadata_indices() {
        let diff_pair = CandidateSet::new(
            "xdp",
            vec![
                join_candidate(
                    DIFF_PAIR_VOUT_COLUMN,
                    DIFF_PAIR_WIDTH_COLUMN,
                    DIFF_PAIR_LENGTH_COLUMN,
                    0.9,
                    4.0e-6,
                    0.4e-6,
                ),
                join_candidate(
                    DIFF_PAIR_VOUT_COLUMN,
                    DIFF_PAIR_WIDTH_COLUMN,
                    DIFF_PAIR_LENGTH_COLUMN,
                    1.0,
                    6.0e-6,
                    0.8e-6,
                ),
            ],
        );
        let current_mirror = CandidateSet::new(
            "xcm",
            vec![
                join_candidate(
                    CURRENT_MIRROR_VOUT_COLUMN,
                    CURRENT_MIRROR_WIDTH_COLUMN,
                    CURRENT_MIRROR_LENGTH_COLUMN,
                    1.0,
                    8.0e-6,
                    1.0e-6,
                ),
                join_candidate(
                    CURRENT_MIRROR_VOUT_COLUMN,
                    CURRENT_MIRROR_WIDTH_COLUMN,
                    CURRENT_MIRROR_LENGTH_COLUMN,
                    1.1,
                    10.0e-6,
                    1.2e-6,
                ),
            ],
        );

        let selections = ota_candidate_pairs(&diff_pair, &current_mirror)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(selections.len(), 1);
        let selection = selections[0];
        assert_eq!((selection.left_index, selection.right_index), (1, 0));
        assert_eq!(
            diff_pair.points[selection.left_index].get(DIFF_PAIR_WIDTH_COLUMN),
            Some(6.0e-6)
        );
        assert_eq!(
            current_mirror.points[selection.right_index].get(CURRENT_MIRROR_LENGTH_COLUMN),
            Some(1.0e-6)
        );
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
        assert_eq!(ANALYSIS_POLICY.mode, AnalysisMode::Prune);
        assert_eq!(
            ANALYSIS_POLICY.targets,
            AnalysisTargets {
                min_dc_gain_db: Some(MIN_DC_GAIN_DB),
                min_bandwidth_3db_hz: Some(MIN_BANDWIDTH_3DB_HZ),
                min_unity_gain_hz: Some(MIN_UNITY_GAIN_HZ),
                min_phase_margin_deg: Some(MIN_PHASE_MARGIN_DEG),
            }
        );
        ANALYSIS_POLICY.validate().unwrap();
        adaptive_ac_config().validate().unwrap();
    }

    #[test]
    fn configures_the_inverting_ota_as_a_negative_transfer() {
        let transfer = ota_transfer_function();

        assert_eq!(transfer.input_node, "VINP");
        assert_eq!(transfer.output_node, "VOUT");
        assert_eq!(transfer.polarity, TransferPolarity::Negative);
    }

    #[test]
    fn classifies_sequential_ac_rejections_and_preserves_count_conservation() {
        let mut counts = AcRejectionCounts::default();
        let mut accepted = 0;

        for completion in [
            AcCompletion::Complete,
            AcCompletion::PrunedAfter(AcMetric::DcGainDb),
            AcCompletion::PrunedAfter(AcMetric::Bandwidth3DbHz),
            AcCompletion::PrunedAfter(AcMetric::UnityGainHz),
            AcCompletion::PrunedAfter(AcMetric::PhaseMarginDeg),
        ] {
            let passed = completion == AcCompletion::Complete;
            if counts.record(completion, passed).unwrap() {
                accepted += 1;
            }
        }

        assert_eq!(
            counts,
            AcRejectionCounts {
                dc_gain: 1,
                bandwidth_3db: 1,
                unity_gain: 1,
                phase_margin: 1,
            }
        );
        assert_eq!(accepted + counts.total(), 5);
        assert!(counts.record(AcCompletion::Complete, false).is_err());
        assert!(
            counts
                .record(AcCompletion::PrunedAfter(AcMetric::DcGainDb), true)
                .is_err()
        );
    }
}
