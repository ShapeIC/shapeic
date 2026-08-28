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
    ElectricalAnalysisExecution, Macro, MacroAcTestbench, MacroAnalysisDomain, MacroCatalog,
    MacroCompactOutputBinding, MacroExecutionConfig, MacroExplorationInput,
    MacroExplorationResult, MacroInterfaceBinding, MacroOutputSource, MacroPort, MacroPortRole,
    PrimitiveInstanceExplorationInput,
};
use shapeic_core::primitive::build::{PrimitiveBuildInput, PrimitiveBuildValue};
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::utils::linspace;
use shapeic_layout::PhysicalLookupTable;
use shapeic_lut::{DeviceLut, LookupTable};

const VOUT_POINTS: usize = 100;
const VBIAS_POINTS: usize = 100;

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

#[derive(Clone, Copy, Debug, PartialEq)]
struct PdkSpec {
    label: &'static str,
    pdk: &'static str,
    nmos_model: &'static str,
    pmos_model: &'static str,
    tail_current: f64,
    mirror_reference: f64,
    vout_start: f64,
    vout_stop: f64,
    vdd: f64,
    vin: f64,
    vbias_start: f64,
    vbias_stop: f64,
    layout_policy: &'static str,
}

const IHP_SPEC: PdkSpec = PdkSpec {
    label: "IHP SG13G2",
    pdk: "ihp-sg13g2",
    nmos_model: "sg13_lv_nmos",
    pmos_model: "sg13_lv_pmos",
    tail_current: 20.0e-6,
    mirror_reference: 1.0,
    vout_start: 0.95,
    vout_stop: 1.1,
    vdd: 1.5,
    vin: 0.9,
    vbias_start: 0.65,
    vbias_stop: 0.79,
    layout_policy: "symmetric-adjacent-with-edge-dummies-v3",
};

const SKY130_SPEC: PdkSpec = PdkSpec {
    label: "SKY130A",
    pdk: "sky130A",
    nmos_model: "sky130_fd_pr__nfet_01v8",
    pmos_model: "sky130_fd_pr__pfet_01v8",
    tail_current: 20.0e-6,
    mirror_reference: 0.6,
    vout_start: 0.8,
    vout_stop: 1.2,
    vdd: 1.8,
    vin: 0.9,
    vbias_start: 0.1,
    vbias_stop: 0.3,
    layout_policy: "symmetric-native-fingers-with-edge-dummies-v1",
};

const GF180_SPEC: PdkSpec = PdkSpec {
    label: "GF180MCU D",
    pdk: "gf180mcuD",
    nmos_model: "nfet_03v3",
    pmos_model: "pfet_03v3",
    tail_current: 20.0e-6,
    mirror_reference: 2.1,
    vout_start: 1.6,
    vout_stop: 2.4,
    vdd: 3.3,
    vin: 1.4,
    vbias_start: 0.3,
    vbias_stop: 0.6,
    layout_policy: "symmetric-native-fingers-with-edge-dummies-v3",
};

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let primitives_dir = manifest.join("../shapeic-cellkit/primitives");
    let testbench_path = manifest.join("examples/ota_4t_v3/gain.spice");
    let options = cli_options()?;

    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(&options.nmos_path)?;
    let pmos_table = LookupTable::open(&options.pmos_path)?;
    let physical_table = options
        .physical_path
        .as_ref()
        .map(PhysicalLookupTable::open)
        .transpose()?;
    let lut_load = stage_start.elapsed();
    let spec = resolve_pdk_spec(&nmos_table, &pmos_table, physical_table.as_ref())?;
    let nmos = nmos_table.model(spec.nmos_model)?;
    let pmos = pmos_table.model(spec.pmos_model)?;

    let primitive_catalog =
        load_primitive_catalog(&primitives_dir).map_err(|error| format!("{error:?}"))?;
    let ota = ota_macro(testbench_path, physical_table.is_some());
    let macro_catalog = MacroCatalog::from_macros([ota.clone()])?;

    let executions = options.execution_configs()?;
    let mut runs = Vec::with_capacity(executions.len());
    for (label, execution) in executions {
        let input = ota_exploration_input(
            &spec,
            nmos,
            pmos,
            physical_table.as_ref(),
            execution,
        )?;
        let result = ota.explore(&primitive_catalog, &macro_catalog, input)?;
        runs.push((label, result));
    }
    validate_comparison_results(&runs)?;
    let (_, result) = runs
        .first()
        .ok_or_else(|| io::Error::other("no OTA exploration was requested"))?;

    write_results_csv(
        result,
        &spec,
        manifest.join("examples/ota_4t_v3/results.csv"),
    )?;

    println!("{} four-transistor OTA exploration", spec.label);
    print_results(result)?;
    print_statistics(result)?;
    println!("LUT load took: {lut_load:?}");
    println!("Total process time: {:?}", total_start.elapsed());
    print_execution_comparison(&runs);
    Ok(())
}

fn ota_exploration_input<'a>(
    spec: &PdkSpec,
    nmos: &'a DeviceLut,
    pmos: &'a DeviceLut,
    physical_table: Option<&'a PhysicalLookupTable>,
    execution: MacroExecutionConfig,
) -> Result<MacroExplorationInput<'a>, Box<dyn Error>> {
    let mut input = MacroExplorationInput::new();
    input.set_execution_config(execution);
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    if let Some(physical_table) = physical_table {
        input.register_physical_lut(physical_table)?;
    }
    input.register_primitive_instance(
        DIFF_PAIR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            diff_pair_input(spec),
            vec![CandidateFilter::at_most(
                DIFF_PAIR_WIDTH_COLUMN,
                MAX_DIFF_PAIR_WIDTH,
            )?],
        ),
    )?;
    input.register_primitive_instance(
        CURRENT_MIRROR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            current_mirror_input(spec),
            vec![CandidateFilter::at_most(
                CURRENT_MIRROR_WIDTH_COLUMN,
                MAX_CURRENT_MIRROR_WIDTH,
            )?],
        ),
    )?;

    Ok(input)
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
    ota.with_compact_output(MacroCompactOutputBinding::new(
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

fn diff_pair_input(spec: &PdkSpec) -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(spec.tail_current),
        ),
        ("VINP".to_owned(), PrimitiveBuildValue::Scalar(spec.vin)),
        (
            "VOUTP".to_owned(),
            PrimitiveBuildValue::Vector(linspace(spec.vout_start, spec.vout_stop, VOUT_POINTS)),
        ),
        (
            "VTAIL".to_owned(),
            PrimitiveBuildValue::Vector(linspace(spec.vbias_start, spec.vbias_stop, VBIAS_POINTS)),
        ),
    ]))
}

fn current_mirror_input(spec: &PdkSpec) -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(spec.tail_current),
        ),
        (
            "VINP".to_owned(),
            PrimitiveBuildValue::Scalar(spec.mirror_reference),
        ),
        (
            "VOUTP".to_owned(),
            PrimitiveBuildValue::Vector(linspace(spec.vout_start, spec.vout_stop, VOUT_POINTS)),
        ),
        ("VDD".to_owned(), PrimitiveBuildValue::Scalar(spec.vdd)),
    ]))
}

fn pdk_spec(pdk: &str) -> Result<PdkSpec, io::Error> {
    match pdk {
        "ihp-sg13g2" => Ok(IHP_SPEC),
        "sky130A" => Ok(SKY130_SPEC),
        "gf180mcuD" => Ok(GF180_SPEC),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported LUT PDK '{pdk}'; expected ihp-sg13g2, sky130A, or gf180mcuD"),
        )),
    }
}

fn validate_pdk_selection(
    nmos_pdk: Option<&str>,
    pmos_pdk: Option<&str>,
    physical: Option<(&str, u32, &str)>,
) -> Result<PdkSpec, io::Error> {
    let nmos_pdk = nmos_pdk.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "NMOS LUT does not declare metadata.pdk",
        )
    })?;
    let pmos_pdk = pmos_pdk.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "PMOS LUT does not declare metadata.pdk",
        )
    })?;
    if pmos_pdk != nmos_pdk {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("NMOS LUT uses PDK '{nmos_pdk}', but PMOS LUT uses '{pmos_pdk}'"),
        ));
    }

    let spec = pdk_spec(nmos_pdk)?;
    if let Some((physical_pdk, format_version, layout_policy)) = physical {
        if physical_pdk != spec.pdk {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "electrical LUTs use PDK '{}', but the physical LUT uses '{physical_pdk}'",
                    spec.pdk
                ),
            ));
        }
        if format_version != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "layout-aware exploration requires physical LUT format v2, found v{format_version}"
                ),
            ));
        }
        if layout_policy != spec.layout_policy {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "physical LUT layout policy '{layout_policy}' does not match '{}' for PDK '{}'",
                    spec.layout_policy, spec.pdk
                ),
            ));
        }
    }
    Ok(spec)
}

fn resolve_pdk_spec(
    nmos: &LookupTable,
    pmos: &LookupTable,
    physical: Option<&PhysicalLookupTable>,
) -> Result<PdkSpec, io::Error> {
    let nmos_pdk = resolve_electrical_lut_pdk(nmos, true)?;
    let pmos_pdk = resolve_electrical_lut_pdk(pmos, false)?;
    validate_pdk_selection(
        Some(nmos_pdk),
        Some(pmos_pdk),
        physical.map(|table| {
            let metadata = table.metadata();
            (
                metadata.pdk.as_str(),
                metadata.format_version,
                metadata.layout_policy.as_str(),
            )
        }),
    )
}

fn resolve_electrical_lut_pdk(table: &LookupTable, nmos: bool) -> Result<&str, io::Error> {
    if let Some(pdk) = table.pdk() {
        return Ok(pdk);
    }

    let inferred = infer_pdk_from_models(table.model_names(), nmos)?;
    Ok(inferred.pdk)
}

fn infer_pdk_from_models<'a>(
    model_names: impl IntoIterator<Item = &'a str>,
    nmos: bool,
) -> Result<PdkSpec, io::Error> {
    let model_names = model_names.into_iter().collect::<Vec<_>>();
    let matching = [IHP_SPEC, SKY130_SPEC, GF180_SPEC]
        .into_iter()
        .filter(|spec| {
            let expected = if nmos {
                spec.nmos_model
            } else {
                spec.pmos_model
            };
            model_names.contains(&expected)
        })
        .collect::<Vec<_>>();

    match matching.as_slice() {
        [spec] => Ok(*spec),
        [] => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} LUT has no metadata.pdk and its models do not identify a supported PDK",
                if nmos { "NMOS" } else { "PMOS" }
            ),
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} LUT has no metadata.pdk and contains models from multiple supported PDKs",
                if nmos { "NMOS" } else { "PMOS" }
            ),
        )),
    }
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
        .ok_or_else(|| {
            io::Error::other(format!(
                "accepted OTA candidate has no '{testbench}' outcome"
            ))
        })
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
    if statistics.testbench(LAYOUT_AWARE_AC_TESTBENCH).is_some() {
        print_testbench_statistics(result, LAYOUT_AWARE_AC_TESTBENCH)?;
    }
    println!("Accepted candidates: {}", statistics.accepted_candidates());
    println!(
        "Total frequency evaluations: {}",
        statistics.frequency_evaluations()
    );
    Ok(())
}

fn validate_comparison_results(
    runs: &[(String, MacroExplorationResult)],
) -> Result<(), io::Error> {
    let Some((baseline_label, baseline)) = runs.first() else {
        return Err(io::Error::other("no exploration result was produced"));
    };
    for (label, result) in &runs[1..] {
        if result.macro_name() != baseline.macro_name()
            || result.candidate_sets() != baseline.candidate_sets()
            || result.accepted() != baseline.accepted()
            || result.statistics() != baseline.statistics()
        {
            return Err(io::Error::other(format!(
                "exploration '{label}' differs from baseline '{baseline_label}'"
            )));
        }
    }
    Ok(())
}

fn print_execution_comparison(runs: &[(String, MacroExplorationResult)]) {
    let baseline_seconds = runs
        .first()
        .map(|(_, result)| result.execution_report().total().as_secs_f64())
        .unwrap_or_default();
    println!("\nMacro execution");
    for (label, result) in runs {
        let report = result.execution_report();
        match report.electrical_execution() {
            ElectricalAnalysisExecution::Sequential => {
                println!("{label}: sequential electrical execution");
            }
            ElectricalAnalysisExecution::Parallel {
                workers,
                batch_size,
            } => println!(
                "{label}: parallel electrical execution, workers={workers}, batch_size={batch_size}"
            ),
        }
        println!(
            "  compatible={}, electrical_survivors={}, accepted={}, batches={}",
            result.statistics().compatible_candidates(),
            report.electrical_survivors(),
            result.statistics().accepted_candidates(),
            report.electrical_batches(),
        );
        println!(
            "  candidate_build={:?}, testbench_preparation={:?}, sequential_evaluation={:?}",
            report.candidate_build(),
            report.testbench_preparation(),
            report.sequential_evaluation(),
        );
        println!(
            "  parallel_electrical={:?}, sequential_tail={:?}, total={:?}",
            report.parallel_electrical_analysis(),
            report.sequential_tail(),
            report.total(),
        );
        let seconds = report.total().as_secs_f64();
        if baseline_seconds > 0.0 && seconds > 0.0 {
            println!("  speedup_vs_first={:.3}x", baseline_seconds / seconds);
        }
    }
    if runs.len() > 1 {
        println!("All comparison runs produced identical candidates, metrics, statistics, and provenance.");
    }
}

fn print_testbench_statistics(
    result: &MacroExplorationResult,
    testbench: &str,
) -> Result<(), io::Error> {
    let statistics = result
        .statistics()
        .testbench(testbench)
        .ok_or_else(|| io::Error::other(format!("OTA result has no '{testbench}' statistics")))?;
    println!(
        "{testbench} evaluated candidates: {}",
        statistics.evaluated_candidates()
    );
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

#[derive(Debug, PartialEq, Eq)]
struct CliOptions {
    nmos_path: PathBuf,
    pmos_path: PathBuf,
    physical_path: Option<PathBuf>,
    electrical_workers: Option<usize>,
    electrical_batch_size: Option<usize>,
    compare_electrical_workers: Vec<usize>,
}

impl CliOptions {
    fn execution_configs(&self) -> Result<Vec<(String, MacroExecutionConfig)>, io::Error> {
        if !self.compare_electrical_workers.is_empty() {
            if self.electrical_workers.is_some() {
                return Err(invalid_input(
                    "--electrical-workers cannot be combined with --compare-electrical-workers",
                ));
            }
            if self.electrical_batch_size.is_some()
                && !self
                    .compare_electrical_workers
                    .iter()
                    .any(|workers| *workers > 1)
            {
                return Err(invalid_input(
                    "--electrical-batch-size requires at least one parallel comparison run",
                ));
            }
            return self
                .compare_electrical_workers
                .iter()
                .copied()
                .map(|workers| {
                    let execution = execution_config(
                        Some(workers),
                        (workers > 1).then_some(self.electrical_batch_size).flatten(),
                    )?;
                    Ok((format!("{workers} worker(s)"), execution))
                })
                .collect();
        }
        let execution = execution_config(self.electrical_workers, self.electrical_batch_size)?;
        Ok(vec![("selected".to_owned(), execution)])
    }
}

fn execution_config(
    workers: Option<usize>,
    batch_size: Option<usize>,
) -> Result<MacroExecutionConfig, io::Error> {
    let mut execution = MacroExecutionConfig::sequential();
    if let Some(workers) = workers {
        execution = execution
            .with_parallel_electrical_analysis(workers)
            .map_err(|error| invalid_input(error.to_string()))?;
    }
    if let Some(batch_size) = batch_size {
        execution = execution
            .with_electrical_batch_size(batch_size)
            .map_err(|error| invalid_input(error.to_string()))?;
    }
    Ok(execution)
}

fn cli_options() -> Result<CliOptions, io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_v3"));
    let usage = || {
        invalid_input(format!(
            "usage: {} <nmos-5d.npz> <pmos-5d.npz> [physical.npz] \
             [--electrical-workers N] [--electrical-batch-size N] \
             [--compare-electrical-workers 1,2,4,8]",
            Path::new(&executable).display()
        ))
    };
    let mut paths = Vec::new();
    let mut electrical_workers = None;
    let mut electrical_batch_size = None;
    let mut compare_electrical_workers = Vec::new();
    while let Some(argument) = arguments.next() {
        if argument == "--electrical-workers" {
            electrical_workers = Some(parse_usize_argument(
                arguments.next().ok_or_else(&usage)?,
                "--electrical-workers",
            )?);
        } else if argument == "--electrical-batch-size" {
            electrical_batch_size = Some(parse_usize_argument(
                arguments.next().ok_or_else(&usage)?,
                "--electrical-batch-size",
            )?);
        } else if argument == "--compare-electrical-workers" {
            let value = arguments.next().ok_or_else(&usage)?;
            let value = value.to_str().ok_or_else(|| {
                invalid_input("--compare-electrical-workers must be valid UTF-8")
            })?;
            compare_electrical_workers = value
                .split(',')
                .map(|worker| parse_usize(worker, "--compare-electrical-workers"))
                .collect::<Result<Vec<_>, _>>()?;
            if compare_electrical_workers.is_empty() {
                return Err(invalid_input(
                    "--compare-electrical-workers requires at least one worker count",
                ));
            }
        } else if argument.to_string_lossy().starts_with("--") {
            return Err(usage());
        } else {
            paths.push(PathBuf::from(argument));
        }
    }
    if !(2..=3).contains(&paths.len()) {
        return Err(usage());
    }
    let physical_path = (paths.len() == 3).then(|| paths.remove(2));
    let pmos_path = paths.remove(1);
    let nmos_path = paths.remove(0);
    let options = CliOptions {
        nmos_path,
        pmos_path,
        physical_path,
        electrical_workers,
        electrical_batch_size,
        compare_electrical_workers,
    };
    options.execution_configs()?;
    Ok(options)
}

fn parse_usize_argument(value: OsString, flag: &str) -> Result<usize, io::Error> {
    let value = value
        .to_str()
        .ok_or_else(|| invalid_input(format!("{flag} must be valid UTF-8")))?;
    parse_usize(value, flag)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, io::Error> {
    value
        .parse::<usize>()
        .map_err(|_| invalid_input(format!("{flag} requires a positive integer, found '{value}'")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shapeic_core::macro_model::{render_expanded_small_signal_netlist, validate_macro};

    #[test]
    fn configures_sequential_and_parallel_execution_modes() {
        assert_eq!(
            execution_config(None, None)
                .unwrap()
                .electrical_analysis(),
            ElectricalAnalysisExecution::Sequential
        );
        assert_eq!(
            execution_config(Some(8), Some(13))
                .unwrap()
                .electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 8,
                batch_size: 13,
            }
        );
        assert!(execution_config(Some(0), None).is_err());
        assert!(execution_config(None, Some(4)).is_err());
        assert!(execution_config(Some(1), Some(4)).is_err());
    }

    #[test]
    fn builds_independent_worker_comparison_runs() {
        let options = CliOptions {
            nmos_path: "nmos.npz".into(),
            pmos_path: "pmos.npz".into(),
            physical_path: None,
            electrical_workers: None,
            electrical_batch_size: Some(5),
            compare_electrical_workers: vec![1, 4, 8],
        };

        let executions = options.execution_configs().unwrap();
        assert_eq!(executions.len(), 3);
        assert_eq!(
            executions[0].1.electrical_analysis(),
            ElectricalAnalysisExecution::Sequential
        );
        assert_eq!(
            executions[1].1.electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 4,
                batch_size: 5,
            }
        );
        assert_eq!(
            executions[2].1.electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 8,
                batch_size: 5,
            }
        );
    }

    #[test]
    fn defines_and_renders_the_complete_typed_ota() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let primitives =
            load_primitive_catalog(&manifest.join("../shapeic-cellkit/primitives")).unwrap();
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
        assert!(
            diff_pair
                .ports()
                .iter()
                .any(|port| { port.physical_port() == "B" && port.node() == "IBIAS" })
        );
        let current_mirror = rendered
            .physical_primitives()
            .iter()
            .find(|physical| physical.instance_path() == CURRENT_MIRROR_INSTANCE)
            .unwrap();
        assert_eq!(current_mirror.lut_primitive(), "currentmirror");
        assert_eq!(current_mirror.candidate_columns().vds(), "vds__xcm__m1");
        assert!(
            current_mirror
                .ports()
                .iter()
                .any(|port| { port.physical_port() == "DREF" && port.node() == "N1" })
        );
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

    #[test]
    fn selects_each_supported_pdk_and_its_cellkit_policy() {
        for expected in [IHP_SPEC, SKY130_SPEC, GF180_SPEC] {
            let selected = validate_pdk_selection(
                Some(expected.pdk),
                Some(expected.pdk),
                Some((expected.pdk, 2, expected.layout_policy)),
            )
            .unwrap();
            assert_eq!(selected, expected);
        }
    }

    #[test]
    fn infers_legacy_lut_pdk_from_a_unique_model() {
        assert_eq!(
            infer_pdk_from_models([IHP_SPEC.nmos_model], true).unwrap(),
            IHP_SPEC
        );
        assert_eq!(
            infer_pdk_from_models([SKY130_SPEC.pmos_model], false).unwrap(),
            SKY130_SPEC
        );
        assert!(infer_pdk_from_models(["unknown"], true).is_err());
        assert!(infer_pdk_from_models([IHP_SPEC.nmos_model, GF180_SPEC.nmos_model], true).is_err());
    }

    #[test]
    fn rejects_incompatible_electrical_and_physical_luts() {
        let mismatched_electrical =
            validate_pdk_selection(Some("sky130A"), Some("gf180mcuD"), None).unwrap_err();
        assert!(mismatched_electrical.to_string().contains("PMOS LUT"));

        let mismatched_physical = validate_pdk_selection(
            Some("sky130A"),
            Some("sky130A"),
            Some(("gf180mcuD", 2, GF180_SPEC.layout_policy)),
        )
        .unwrap_err();
        assert!(mismatched_physical.to_string().contains("physical LUT"));

        let legacy_physical = validate_pdk_selection(
            Some("ihp-sg13g2"),
            Some("ihp-sg13g2"),
            Some(("ihp-sg13g2", 1, IHP_SPEC.layout_policy)),
        )
        .unwrap_err();
        assert!(legacy_physical.to_string().contains("format v2"));

        let wrong_policy = validate_pdk_selection(
            Some("sky130A"),
            Some("sky130A"),
            Some(("sky130A", 2, GF180_SPEC.layout_policy)),
        )
        .unwrap_err();
        assert!(wrong_policy.to_string().contains("layout policy"));
    }
}

use std::fs::File;
use std::io::{BufWriter, Write};

fn write_results_csv(
    result: &MacroExplorationResult,
    spec: &PdkSpec,
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

        let dp = |column| selected_value(result, accepted, DIFF_PAIR_INSTANCE, column);
        let cm = |column| selected_value(result, accepted, CURRENT_MIRROR_INSTANCE, column);

        let electrical =
            required_metrics(result_metrics(result, accepted, ELECTRICAL_AC_TESTBENCH)?)?;

        write!(
            writer,
            "{candidate_id},{dp_index},{cm_index},\
             {:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.0},{:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.0},{:.17e},{:.17e},{:.17e},\
             {:.17e},{:.17e},{:.17e},{:.17e}",
            spec.vin,
            spec.vdd,
            spec.tail_current,
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
            let layout =
                required_metrics(result_metrics(result, accepted, LAYOUT_AWARE_AC_TESTBENCH)?)?;
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
