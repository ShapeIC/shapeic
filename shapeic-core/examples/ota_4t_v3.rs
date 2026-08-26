use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shapeic_core::analysis::{
    AcMetric, AcMetricSet, AcMetrics, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode,
    AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::exploration::filter::CandidateFilter;
use shapeic_core::macro_model::{
    Macro, MacroAcTestbench, MacroAnalysisDomain, MacroCatalog, MacroCompactOutputBinding,
    MacroExplorationInput, MacroExplorationResult, MacroInterfaceBinding, MacroOutputSource,
    MacroPort, MacroPortRole, PrimitiveInstanceExplorationInput,
};
use shapeic_core::primitive::build::{PrimitiveBuildInput, PrimitiveBuildValue};
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::utils::linspace;
use shapeic_lut::LookupTable;
use shapeic_layout::PhysicalLookupTable;

const TAIL_CURRENT: f64 = 20.0e-6;
const VOUT: f64 = 1.0;
const VOUT_START: f64 = 0.95;
const VOUT_STOP: f64 = 1.1;
const VOUT_POINTS: usize = 10;
const VDD: f64 = 1.5;
const VIN: f64 = 0.9;
const VBIAS_START: f64 = 0.65;
const VBIAS_STOP: f64 = 0.79;
const VBIAS_POINTS: usize = 10;

const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";
const ELECTRICAL_AC_TESTBENCH: &str = "gain_electrical";
const LAYOUT_AWARE_AC_TESTBENCH: &str = "gain_layout_aware";

const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
const AC_COARSE_POINTS_PER_DECADE: usize = 4;
const AC_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
const AC_MAX_REFINEMENT_STEPS: usize = 32;
const MIN_DC_GAIN_DB: f64 = 25.0;
const MIN_BANDWIDTH_3DB_HZ: f64 = 1.0e6;
const MIN_UNITY_GAIN_HZ: f64 = 1.0e7;
const MIN_PHASE_MARGIN_DEG: f64 = 45.0;

const DIFF_PAIR_GM_COLUMN: &str = "gm__xdp__m1";
const DIFF_PAIR_RO_COLUMN: &str = "ro__xdp__m1";
const CURRENT_MIRROR_GM_COLUMN: &str = "gm__xcm__m1";
const CURRENT_MIRROR_RO_COLUMN: &str = "ro__xcm__m1";
const DIFF_PAIR_VIN_COLUMN: &str = "xdp.vinp";
const DIFF_PAIR_VOUT_COLUMN: &str = "xdp.voutp";
const DIFF_PAIR_VBIAS_COLUMN: &str = "xdp.vtail";
const CURRENT_MIRROR_VDD_COLUMN: &str = "xcm.vdd";
const CURRENT_MIRROR_VOUT_COLUMN: &str = "xcm.voutp";
const DIFF_PAIR_WIDTH_COLUMN: &str = "width__xdp__m1";
const DIFF_PAIR_FINGER_WIDTH_COLUMN: &str = "finger_width__xdp__m1";
const DIFF_PAIR_LENGTH_COLUMN: &str = "length__xdp__m1";
const DIFF_PAIR_NF_COLUMN: &str = "nf__xdp__m1";
const DIFF_PAIR_VBS_COLUMN: &str = "vbs__xdp__m1";
const DIFF_PAIR_VGS_COLUMN: &str = "vgs__xdp__m1";
const DIFF_PAIR_VDS_COLUMN: &str = "vds__xdp__m1";
const CURRENT_MIRROR_WIDTH_COLUMN: &str = "width__xcm__m1";
const CURRENT_MIRROR_FINGER_WIDTH_COLUMN: &str = "finger_width__xcm__m1";
const CURRENT_MIRROR_LENGTH_COLUMN: &str = "length__xcm__m1";
const CURRENT_MIRROR_NF_COLUMN: &str = "nf__xcm__m1";
const CURRENT_MIRROR_VBS_COLUMN: &str = "vbs__xcm__m1";
const CURRENT_MIRROR_VGS_COLUMN: &str = "vgs__xcm__m1";
const CURRENT_MIRROR_VDS_COLUMN: &str = "vds__xcm__m1";
const MAX_DIFF_PAIR_WIDTH: f64 = 100.0e-6;
const MAX_CURRENT_MIRROR_WIDTH: f64 = 100.0e-6;

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let primitives_dir = manifest.join("../shapeic-cellkit/primitives");
    let testbench_path = manifest.join("examples/ota_4t_v3/gain.spice");
    let (nmos_path, pmos_path, physical_path) = lut_paths()?;

    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let physical_table = physical_path
        .map(PhysicalLookupTable::open)
        .transpose()?;
    let lut_load = stage_start.elapsed();
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;

    let primitive_catalog =
        load_primitive_catalog(&primitives_dir).map_err(|error| format!("{error:?}"))?;
    let ota = ota_macro(testbench_path, physical_table.is_some());
    let macro_catalog = MacroCatalog::from_macros([ota.clone()])?;

    let mut input = MacroExplorationInput::new();
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    if let Some(physical_table) = &physical_table {
        input.register_physical_lut(physical_table)?;
    }
    input.register_primitive_instance(
        DIFF_PAIR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            diff_pair_input(),
            vec![CandidateFilter::at_most(
                DIFF_PAIR_WIDTH_COLUMN,
                MAX_DIFF_PAIR_WIDTH,
            )?],
        ),
    )?;
    input.register_primitive_instance(
        CURRENT_MIRROR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            current_mirror_input(),
            vec![CandidateFilter::at_most(
                CURRENT_MIRROR_WIDTH_COLUMN,
                MAX_CURRENT_MIRROR_WIDTH,
            )?],
        ),
    )?;

    let stage_start = Instant::now();
    let result = ota.explore(&primitive_catalog, &macro_catalog, input)?;
    let exploration_time = stage_start.elapsed();
    
    write_results_csv(&result, manifest.join("examples/ota_4t_v3/results.csv"))?;

    print_results(&result)?;
    print_statistics(&result)?;
    println!("LUT load took: {lut_load:?}");
    println!("Macro exploration took: {exploration_time:?}");
    println!("Total time: {:?}", total_start.elapsed());
    Ok(())
}

fn ota_macro(testbench_path: PathBuf, layout_aware: bool) -> Macro {
    let circuit = Circuit::builder()
        .primitive(
            DIFF_PAIR_INSTANCE,
            "simplediffpair",
            [
                ("VINP", "VINP"),
                ("VINN", "VINN"),
                ("VOUTP", "VOUT"),
                ("VOUTN", "N1"),
                ("VTAIL", "IBIAS"),
                ("VSS", "IBIAS"),
            ],
        )
        .primitive(
            CURRENT_MIRROR_INSTANCE,
            "simplecurrentmirror",
            [("VINP", "N1"), ("VOUTP", "VOUT"), ("VDD", "VDD")],
        )
        .build();

    let compact_model = Circuit::builder()
        .vccs("gm_dp_m1", "VOUT", "IBIAS", "VINP", "IBIAS", "gm_dp")
        .resistor("ro_dp_m1", "VOUT", "IBIAS", "ro_dp")
        .vccs("gm_dp_m2", "N1", "IBIAS", "VINN", "IBIAS", "gm_dp")
        .resistor("ro_dp_m2", "N1", "IBIAS", "ro_dp")
        .vccs("gm_cm_m1", "VOUT", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m1", "VOUT", "VDD", "ro_cm")
        .vccs("gm_cm_m2", "N1", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m2", "N1", "VDD", "ro_cm")
        .build();

    let mut ota = Macro::new(
        "ota_4t",
        vec![
            MacroPort::new("VINP", MacroPortRole::Input),
            MacroPort::new("VINN", MacroPortRole::Input),
            MacroPort::new("VOUT", MacroPortRole::Output),
            MacroPort::new("IBIAS", MacroPortRole::Bias),
            MacroPort::new("VDD", MacroPortRole::Supply),
        ],
        circuit,
        compact_model,
    )
    .with_ac_testbench(MacroAcTestbench::from_spice_file(
        ELECTRICAL_AC_TESTBENCH,
        &testbench_path,
        ota_ac_analysis(),
    ));
    if layout_aware {
        ota = ota.with_ac_testbench(
            MacroAcTestbench::from_spice_file(
                LAYOUT_AWARE_AC_TESTBENCH,
                testbench_path,
                ota_ac_analysis(),
            )
            .with_domain(MacroAnalysisDomain::LayoutAware),
        );
    }
    ota
    .with_compact_output(MacroCompactOutputBinding::new(
        "gm_dp",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_GM_COLUMN),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "ro_dp",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_RO_COLUMN),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "gm_cm",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CURRENT_MIRROR_GM_COLUMN),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "ro_cm",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CURRENT_MIRROR_RO_COLUMN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VINP",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_VIN_COLUMN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VINN",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_VIN_COLUMN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VOUT",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_VOUT_COLUMN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "IBIAS",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DIFF_PAIR_VBIAS_COLUMN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VDD",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CURRENT_MIRROR_VDD_COLUMN),
    ))
}

fn ota_ac_analysis() -> AcAnalysis {
    AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT").with_polarity(TransferPolarity::Negative),
        AdaptiveAcConfig {
            min_frequency_hz: AC_MIN_HZ,
            max_frequency_hz: AC_MAX_HZ,
            coarse_points_per_decade: AC_COARSE_POINTS_PER_DECADE,
            crossing_relative_tolerance: AC_CROSSING_RELATIVE_TOLERANCE,
            max_refinement_steps: AC_MAX_REFINEMENT_STEPS,
            retain_samples: false,
        },
        AdaptiveAcPolicy {
            mode: AnalysisMode::Prune,
            targets: AnalysisTargets {
                min_dc_gain_db: Some(MIN_DC_GAIN_DB),
                min_bandwidth_3db_hz: Some(MIN_BANDWIDTH_3DB_HZ),
                min_unity_gain_hz: Some(MIN_UNITY_GAIN_HZ),
                min_phase_margin_deg: Some(MIN_PHASE_MARGIN_DEG),
            },
            metrics: AcMetricSet::ALL,
        },
    )
}

fn diff_pair_input() -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(TAIL_CURRENT),
        ),
        ("VINP".to_owned(), PrimitiveBuildValue::Scalar(VIN)),
        (
            "VOUTP".to_owned(),
            PrimitiveBuildValue::Vector(linspace(VOUT_START, VOUT_STOP, VOUT_POINTS)),
        ),
        (
            "VTAIL".to_owned(),
            PrimitiveBuildValue::Vector(linspace(VBIAS_START, VBIAS_STOP, VBIAS_POINTS)),
        ),
    ]))
}

fn current_mirror_input() -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(TAIL_CURRENT),
        ),
        ("VINP".to_owned(), PrimitiveBuildValue::Scalar(VOUT)),
        (
            "VOUTP".to_owned(),
            PrimitiveBuildValue::Vector(linspace(VOUT_START, VOUT_STOP, VOUT_POINTS)),
        ),
        ("VDD".to_owned(), PrimitiveBuildValue::Scalar(VDD)),
    ]))
}

fn print_results(result: &MacroExplorationResult) -> Result<(), io::Error> {
    let has_layout_aware = result
        .statistics()
        .testbench(LAYOUT_AWARE_AC_TESTBENCH)
        .is_some();

    println!("\nOTA sizing and operating points");
    println!(
        "{:<8} {:<8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>5} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>5} {:>8} {:>8} {:>8}",
        "dp_idx",
        "cm_idx",
        "vout",
        "vbias",
        "dp_w_um",
        "dp_wf_um",
        "dp_l_um",
        "dp_nf",
        "dp_vbs",
        "dp_vgs",
        "dp_vds",
        "cm_w_um",
        "cm_wf_um",
        "cm_l_um",
        "cm_nf",
        "cm_vbs",
        "cm_vgs",
        "cm_vds",
    );

    for accepted in result.accepted() {
        let dp_index = selected_index(result, accepted, DIFF_PAIR_INSTANCE)?;
        let cm_index = selected_index(result, accepted, CURRENT_MIRROR_INSTANCE)?;
        let dp_value = |column| selected_value(result, accepted, DIFF_PAIR_INSTANCE, column);
        let cm_value = |column| selected_value(result, accepted, CURRENT_MIRROR_INSTANCE, column);
        let vout = dp_value(DIFF_PAIR_VOUT_COLUMN)?;
        debug_assert_eq!(vout, cm_value(CURRENT_MIRROR_VOUT_COLUMN)?);

        println!(
            "{:<8} {:<8} {:>8.4} {:>8.4} {:>9.4} {:>9.4} {:>8.4} {:>5.0} {:>8.4} {:>8.4} {:>8.4} {:>9.4} {:>9.4} {:>8.4} {:>5.0} {:>8.4} {:>8.4} {:>8.4}",
            dp_index,
            cm_index,
            vout,
            dp_value(DIFF_PAIR_VBIAS_COLUMN)?,
            dp_value(DIFF_PAIR_WIDTH_COLUMN)? * 1.0e6,
            dp_value(DIFF_PAIR_FINGER_WIDTH_COLUMN)? * 1.0e6,
            dp_value(DIFF_PAIR_LENGTH_COLUMN)? * 1.0e6,
            dp_value(DIFF_PAIR_NF_COLUMN)?,
            dp_value(DIFF_PAIR_VBS_COLUMN)?,
            dp_value(DIFF_PAIR_VGS_COLUMN)?,
            dp_value(DIFF_PAIR_VDS_COLUMN)?,
            cm_value(CURRENT_MIRROR_WIDTH_COLUMN)? * 1.0e6,
            cm_value(CURRENT_MIRROR_FINGER_WIDTH_COLUMN)? * 1.0e6,
            cm_value(CURRENT_MIRROR_LENGTH_COLUMN)? * 1.0e6,
            cm_value(CURRENT_MIRROR_NF_COLUMN)?,
            cm_value(CURRENT_MIRROR_VBS_COLUMN)?,
            cm_value(CURRENT_MIRROR_VGS_COLUMN)?,
            cm_value(CURRENT_MIRROR_VDS_COLUMN)?,
        );
    }

    println!("\nOTA AC results");
    if has_layout_aware {
        println!(
            "{:<8} {:<8} {:>11} {:>12} {:>12} {:>10} {:>11} {:>12} {:>12} {:>10} {:>11} {:>11} {:>11} {:>11}",
            "dp_idx",
            "cm_idx",
            "e_gain_db",
            "e_f3db_hz",
            "e_ugf_hz",
            "e_pm_deg",
            "la_gain_db",
            "la_f3db_hz",
            "la_ugf_hz",
            "la_pm_deg",
            "d_gain_db",
            "d_f3db_%",
            "d_ugf_%",
            "d_pm_deg",
        );
    } else {
        println!(
            "{:<8} {:<8} {:>12} {:>12} {:>12} {:>10}",
            "dp_idx", "cm_idx", "dc_gain_db", "f3db_hz", "ugf_hz", "pm_deg",
        );
    }

    for accepted in result.accepted() {
        let dp_index = selected_index(result, accepted, DIFF_PAIR_INSTANCE)?;
        let cm_index = selected_index(result, accepted, CURRENT_MIRROR_INSTANCE)?;
        let electrical = result_metrics(result, accepted, ELECTRICAL_AC_TESTBENCH)?;
        let electrical_values = required_metrics(electrical)?;
        if has_layout_aware {
            let layout = result_metrics(result, accepted, LAYOUT_AWARE_AC_TESTBENCH)?;
            let layout_values = required_metrics(layout)?;
            println!(
                "{:<8} {:<8} {:>11.6} {:>12.6e} {:>12.6e} {:>10.6} {:>11.6} {:>12.6e} {:>12.6e} {:>10.6} {:>11.6} {:>11.6} {:>11.6} {:>11.6}",
                dp_index,
                cm_index,
                electrical_values[0],
                electrical_values[1],
                electrical_values[2],
                electrical_values[3],
                layout_values[0],
                layout_values[1],
                layout_values[2],
                layout_values[3],
                layout_values[0] - electrical_values[0],
                relative_difference_percent(layout_values[1], electrical_values[1]),
                relative_difference_percent(layout_values[2], electrical_values[2]),
                layout_values[3] - electrical_values[3],
            );
        } else {
            println!(
                "{:<8} {:<8} {:>12.6} {:>12.6e} {:>12.6e} {:>10.6}",
                dp_index,
                cm_index,
                electrical_values[0],
                electrical_values[1],
                electrical_values[2],
                electrical_values[3],
            );
        }
    }
    Ok(())
}

fn result_metrics<'a>(
    result: &MacroExplorationResult,
    accepted: &'a shapeic_core::macro_model::MacroAcceptedCandidate,
    testbench: &str,
) -> Result<&'a AcMetrics, io::Error> {
    result
        .ac_outcome(accepted, testbench)
        .map(|outcome| &outcome.metrics)
        .ok_or_else(|| io::Error::other(format!("accepted OTA candidate has no '{testbench}' outcome")))
}

fn required_metrics(metrics: &AcMetrics) -> Result<[f64; 4], io::Error> {
    Ok([
        required_metric(metrics.dc_gain_db, "DC gain")?,
        required_metric(metrics.bandwidth_3db_hz, "bandwidth")?,
        required_metric(metrics.unity_gain_hz, "UGF")?,
        required_metric(metrics.phase_margin_deg, "phase margin")?,
    ])
}

fn relative_difference_percent(value: f64, reference: f64) -> f64 {
    (value - reference) / reference.abs() * 100.0
}

fn print_statistics(result: &MacroExplorationResult) -> Result<(), io::Error> {
    let diff_pair = result
        .candidate_sets()
        .instance(DIFF_PAIR_INSTANCE)
        .ok_or_else(|| io::Error::other("OTA result has no diff-pair candidate set"))?;
    let current_mirror = result
        .candidate_sets()
        .instance(CURRENT_MIRROR_INSTANCE)
        .ok_or_else(|| io::Error::other("OTA result has no current-mirror candidate set"))?;
    let possible_pairs = diff_pair
        .candidates()
        .points
        .len()
        .checked_mul(current_mirror.candidates().points.len())
        .ok_or_else(|| io::Error::other("OTA candidate pair count overflows usize"))?;
    let statistics = result.statistics();

    println!(
        "Diff-pair candidates rejected by width: {}",
        diff_pair.filter_report().rejected_count()
    );
    println!(
        "Current-mirror candidates rejected by width: {}",
        current_mirror.filter_report().rejected_count()
    );
    println!("Candidate pairs: {possible_pairs}");
    println!(
        "Compatible candidate pairs: {}",
        statistics.compatible_candidates()
    );
    println!(
        "Rejected by shared VOUT: {}",
        possible_pairs - statistics.compatible_candidates()
    );
    print_testbench_statistics(result, ELECTRICAL_AC_TESTBENCH)?;
    if statistics
        .testbench(LAYOUT_AWARE_AC_TESTBENCH)
        .is_some()
    {
        print_testbench_statistics(result, LAYOUT_AWARE_AC_TESTBENCH)?;
    }
    println!("Accepted candidates: {}", statistics.accepted_candidates());
    println!(
        "Total frequency evaluations: {}",
        statistics.frequency_evaluations()
    );
    Ok(())
}

fn print_testbench_statistics(
    result: &MacroExplorationResult,
    testbench: &str,
) -> Result<(), io::Error> {
    let statistics = result
        .statistics()
        .testbench(testbench)
        .ok_or_else(|| io::Error::other(format!("OTA result has no '{testbench}' statistics")))?;
    println!("{testbench} evaluated candidates: {}", statistics.evaluated_candidates());
    println!(
        "{testbench} rejected outside physical domain: {}",
        statistics.physical_domain_rejections()
    );
    for (metric, label) in [
        (AcMetric::DcGainDb, "DC gain"),
        (AcMetric::Bandwidth3DbHz, "3 dB bandwidth"),
        (AcMetric::UnityGainHz, "UGF"),
        (AcMetric::PhaseMarginDeg, "phase margin"),
    ] {
        println!(
            "{testbench} rejected by {label}: {}",
            statistics.rejections().count(metric)
        );
    }
    println!(
        "{testbench} frequency evaluations: {}",
        statistics.frequency_evaluations()
    );
    Ok(())
}

fn selected_index(
    result: &MacroExplorationResult,
    accepted: &shapeic_core::macro_model::MacroAcceptedCandidate,
    instance: &str,
) -> Result<usize, io::Error> {
    result
        .selected_candidate_index(accepted, instance)
        .ok_or_else(|| io::Error::other(format!("accepted candidate does not select '{instance}'")))
}

fn selected_value(
    result: &MacroExplorationResult,
    accepted: &shapeic_core::macro_model::MacroAcceptedCandidate,
    instance: &str,
    column: &str,
) -> Result<f64, io::Error> {
    result
        .selected_candidate(accepted, instance)
        .and_then(|candidate| candidate.get(column))
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            io::Error::other(format!(
                "accepted candidate for '{instance}' has no finite value for '{column}'"
            ))
        })
}

fn lut_paths() -> Result<(PathBuf, PathBuf, Option<PathBuf>), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_v3"));
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
    use super::*;
    use shapeic_core::macro_model::{render_expanded_small_signal_netlist, validate_macro};

    #[test]
    fn defines_and_renders_the_complete_typed_ota() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let primitives =
            load_primitive_catalog(&manifest.join("../analoglib/primitives")).unwrap();
        let ota = ota_macro(manifest.join("examples/ota_4t_v3/gain.spice"), true);
        let macros = MacroCatalog::from_macros([ota.clone()]).unwrap();

        let [electrical, layout_aware] = ota.exploration().testbenches() else {
            panic!("layout-aware OTA must contain two ordered AC testbenches");
        };
        assert_eq!(electrical.name(), ELECTRICAL_AC_TESTBENCH);
        assert_eq!(electrical.domain(), MacroAnalysisDomain::Electrical);
        assert_eq!(layout_aware.name(), LAYOUT_AWARE_AC_TESTBENCH);
        assert_eq!(layout_aware.domain(), MacroAnalysisDomain::LayoutAware);
        assert_eq!(electrical.analysis(), layout_aware.analysis());

        let errors = validate_macro(&ota, &primitives, &macros);
        assert!(errors.is_empty(), "invalid OTA macro: {errors:?}");

        let rendered = render_expanded_small_signal_netlist(&ota, &primitives, &macros).unwrap();
        assert_eq!(rendered.primitive_branches().len(), 4);
        assert_eq!(rendered.parameter_order().len(), 8);
        assert!(rendered.primitive_branches().iter().any(|branch| {
            branch.instance_path() == DIFF_PAIR_INSTANCE
                && branch.branch_name() == "m1"
                && branch.gate_node() == "VINP"
                && branch.drain_node() == "VOUT"
                && branch.source_node() == "IBIAS"
                && branch.bulk_node() == "IBIAS"
        }));
        assert!(rendered.primitive_branches().iter().any(|branch| {
            branch.instance_path() == CURRENT_MIRROR_INSTANCE
                && branch.branch_name() == "m2"
                && branch.gate_node() == "N1"
                && branch.drain_node() == "N1"
                && branch.source_node() == "VDD"
                && branch.bulk_node() == "VDD"
        }));
        assert_eq!(rendered.physical_primitives().len(), 2);
        let diff_pair = rendered
            .physical_primitives()
            .iter()
            .find(|physical| physical.instance_path() == DIFF_PAIR_INSTANCE)
            .unwrap();
        assert_eq!(diff_pair.lut_primitive(), "simplediffpair");
        assert_eq!(diff_pair.candidate_columns().vgs(), "vgs__xdp__m1");
        assert!(diff_pair.ports().iter().any(|port| {
            port.physical_port() == "B" && port.node() == "IBIAS"
        }));
        let current_mirror = rendered
            .physical_primitives()
            .iter()
            .find(|physical| physical.instance_path() == CURRENT_MIRROR_INSTANCE)
            .unwrap();
        assert_eq!(current_mirror.lut_primitive(), "currentmirror");
        assert_eq!(current_mirror.candidate_columns().vds(), "vds__xcm__m1");
        assert!(current_mirror.ports().iter().any(|port| {
            port.physical_port() == "DREF" && port.node() == "N1"
        }));
    }

    #[test]
    fn keeps_the_default_flow_electrical_without_a_physical_lut() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ota = ota_macro(manifest.join("examples/ota_4t_v3/gain.spice"), false);
        let [electrical] = ota.exploration().testbenches() else {
            panic!("electrical OTA must contain exactly one AC testbench");
        };

        assert_eq!(electrical.name(), ELECTRICAL_AC_TESTBENCH);
        assert_eq!(electrical.domain(), MacroAnalysisDomain::Electrical);
    }
}

use std::fs::File;
use std::io::{BufWriter, Write};

fn write_results_csv(
    result: &MacroExplorationResult,
    path: impl AsRef<Path>,
) -> Result<(), io::Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    let has_layout_aware = result
        .statistics()
        .testbench(LAYOUT_AWARE_AC_TESTBENCH)
        .is_some();

    write!(
        writer,
        "candidate_id,dp_index,cm_index,vin_v,vdd_v,tail_current_a,\
         vout_v,vbias_v,n1_v,\
         dp_width_m,dp_finger_width_m,dp_length_m,dp_nf,dp_vbs_v,dp_vgs_v,dp_vds_v,\
         cm_width_m,cm_finger_width_m,cm_length_m,cm_nf,cm_vbs_v,cm_vgs_v,cm_vds_v,\
         electrical_dc_gain_db,electrical_bandwidth_3db_hz,\
         electrical_unity_gain_hz,electrical_phase_margin_deg"
    )?;
    if has_layout_aware {
        write!(
            writer,
            ",layout_dc_gain_db,layout_bandwidth_3db_hz,layout_unity_gain_hz,\
             layout_phase_margin_deg,delta_dc_gain_db,delta_bandwidth_3db_percent,\
             delta_unity_gain_percent,delta_phase_margin_deg"
        )?;
    }
    writeln!(writer)?;

    for (candidate_id, accepted) in result.accepted().iter().enumerate() {
        let dp_index = selected_index(result, accepted, DIFF_PAIR_INSTANCE)?;
        let cm_index = selected_index(result, accepted, CURRENT_MIRROR_INSTANCE)?;

        let dp = |column| {
            selected_value(result, accepted, DIFF_PAIR_INSTANCE, column)
        };
        let cm = |column| {
            selected_value(result, accepted, CURRENT_MIRROR_INSTANCE, column)
        };

        let electrical = required_metrics(result_metrics(
            result,
            accepted,
            ELECTRICAL_AC_TESTBENCH,
        )?)?;

        write!(
            writer,
            "{candidate_id},{dp_index},{cm_index},\
             {VIN:.17e},{VDD:.17e},{TAIL_CURRENT:.17e},\
             {:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.0},{:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.0},{:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.17e}",
            dp(DIFF_PAIR_VOUT_COLUMN)?,
            dp(DIFF_PAIR_VBIAS_COLUMN)?,
            cm("xcm.vinp")?,
            dp(DIFF_PAIR_WIDTH_COLUMN)?,
            dp(DIFF_PAIR_FINGER_WIDTH_COLUMN)?,
            dp(DIFF_PAIR_LENGTH_COLUMN)?,
            dp(DIFF_PAIR_NF_COLUMN)?,
            dp(DIFF_PAIR_VBS_COLUMN)?,
            dp(DIFF_PAIR_VGS_COLUMN)?,
            dp(DIFF_PAIR_VDS_COLUMN)?,
            cm(CURRENT_MIRROR_WIDTH_COLUMN)?,
            cm(CURRENT_MIRROR_FINGER_WIDTH_COLUMN)?,
            cm(CURRENT_MIRROR_LENGTH_COLUMN)?,
            cm(CURRENT_MIRROR_NF_COLUMN)?,
            cm(CURRENT_MIRROR_VBS_COLUMN)?,
            cm(CURRENT_MIRROR_VGS_COLUMN)?,
            cm(CURRENT_MIRROR_VDS_COLUMN)?,
            electrical[0],
            electrical[1],
            electrical[2],
            electrical[3],
        )?;
        if has_layout_aware {
            let layout = required_metrics(result_metrics(
                result,
                accepted,
                LAYOUT_AWARE_AC_TESTBENCH,
            )?)?;
            write!(
                writer,
                ",{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                layout[0],
                layout[1],
                layout[2],
                layout[3],
                layout[0] - electrical[0],
                relative_difference_percent(layout[1], electrical[1]),
                relative_difference_percent(layout[2], electrical[2]),
                layout[3] - electrical[3],
            )?;
        }
        writeln!(writer)?;
    }

    writer.flush()
}

fn required_metric(value: Option<f64>, name: &str) -> Result<f64, io::Error> {
    value.ok_or_else(|| io::Error::other(format!("missing {name}")))
}
