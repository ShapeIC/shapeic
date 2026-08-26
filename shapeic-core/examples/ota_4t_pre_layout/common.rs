use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use shapeic_core::analysis::{
    AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::exploration::filter::CandidateFilter;
use shapeic_core::macro_model::{
    Macro, MacroAcTestbench, MacroAcceptedCandidate, MacroCatalog, MacroCompactOutputBinding,
    MacroExplorationInput, MacroExplorationResult, MacroInterfaceBinding, MacroOutputSource,
    MacroPort, MacroPortRole, PrimitiveInstanceExplorationInput,
};
use shapeic_core::primitive::build::{PrimitiveBuildInput, PrimitiveBuildValue};
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::utils::linspace;
use shapeic_lut::verification::{
    VerificationConfig, VerificationEngine, VerificationInput, VerificationReport,
    VerificationStatus,
};
use shapeic_lut::{LookupTable, OperatingPoint};

const OPEN_PDKS_REVISION: &str = "026824c7969ce6f4fc9678e6ca04b0a06a596c4b";
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";
const AC_TESTBENCH: &str = "gain_electrical";
const MAX_TOTAL_WIDTH: f64 = 200.0e-6;
const MAX_FINGER_WIDTH: f64 = 10.0e-6;
const OUTPUT_BIAS_INDUCTANCE_H: f64 = 1.0e9;
const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
const AC_COARSE_POINTS_PER_DECADE: usize = 4;
const AC_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
const AC_MAX_REFINEMENT_STEPS: usize = 32;
const NGSPICE_POINTS_PER_DECADE: usize = 100;

const DP_GM: &str = "gm__xdp__m1";
const DP_RO: &str = "ro__xdp__m1";
const CM_GM: &str = "gm__xcm__m1";
const CM_RO: &str = "ro__xcm__m1";
const DP_VIN: &str = "xdp.vinp";
const DP_VOUT: &str = "xdp.voutp";
const DP_VTAIL: &str = "xdp.vtail";
const CM_VDD: &str = "xcm.vdd";
const CM_VREF: &str = "xcm.vinp";
const CM_VOUT: &str = "xcm.voutp";

const DP_COLUMNS: DeviceColumns = DeviceColumns {
    width: "width__xdp__m1",
    finger_width: "finger_width__xdp__m1",
    length: "length__xdp__m1",
    nf: "nf__xdp__m1",
    vbs: "vbs__xdp__m1",
    vgs: "vgs__xdp__m1",
    vds: "vds__xdp__m1",
};
const CM_COLUMNS: DeviceColumns = DeviceColumns {
    width: "width__xcm__m1",
    finger_width: "finger_width__xcm__m1",
    length: "length__xcm__m1",
    nf: "nf__xcm__m1",
    vbs: "vbs__xcm__m1",
    vgs: "vgs__xcm__m1",
    vds: "vds__xcm__m1",
};

const VERIFICATION_DYNAMIC_VARIABLES: [&str; 10] = [
    "dp_length",
    "dp_width",
    "dp_nf",
    "cm_length",
    "cm_width",
    "cm_nf",
    "tail_current",
    "vdd_dc",
    "vin_dc",
    "vout_dc",
];

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PdkFlavor {
    Sky130,
    Gf180,
}

#[derive(Clone, Copy, Debug)]
pub struct OtaPreLayoutSpec {
    label: &'static str,
    pdk: &'static str,
    revision: &'static str,
    corner: &'static str,
    temperature_c: f64,
    nominal_voltage: f64,
    nmos_model: &'static str,
    pmos_model: &'static str,
    tail_current: f64,
    vin: f64,
    vdd: f64,
    mirror_reference: f64,
    vout_start: f64,
    vout_stop: f64,
    vtail_start: f64,
    vtail_stop: f64,
    geometry_unit_m: f64,
    template_file: &'static str,
    output_directory: &'static str,
    flavor: PdkFlavor,
}

#[allow(dead_code)]
pub const fn sky130_spec() -> OtaPreLayoutSpec {
    OtaPreLayoutSpec {
        label: "SKY130A",
        pdk: "sky130A",
        revision: OPEN_PDKS_REVISION,
        corner: "tt",
        temperature_c: 27.0,
        nominal_voltage: 1.8,
        nmos_model: "sky130_fd_pr__nfet_01v8",
        pmos_model: "sky130_fd_pr__pfet_01v8",
        tail_current: 20.0e-6,
        vin: 0.9,
        vdd: 1.8,
        mirror_reference: 0.6,
        vout_start: 0.8,
        vout_stop: 1.2,
        vtail_start: 0.1,
        vtail_stop: 0.3,
        geometry_unit_m: 1.0e-6,
        template_file: "sky130A.spice",
        output_directory: "shapeic-ota-4t-sky130-pre-layout",
        flavor: PdkFlavor::Sky130,
    }
}

#[allow(dead_code)]
pub const fn gf180_spec() -> OtaPreLayoutSpec {
    OtaPreLayoutSpec {
        label: "GF180MCU D",
        pdk: "gf180mcuD",
        revision: OPEN_PDKS_REVISION,
        corner: "typical",
        temperature_c: 25.0,
        nominal_voltage: 3.3,
        nmos_model: "nfet_03v3",
        pmos_model: "pfet_03v3",
        tail_current: 20.0e-6,
        vin: 1.4,
        vdd: 3.3,
        mirror_reference: 2.1,
        vout_start: 1.6,
        vout_stop: 2.4,
        vtail_start: 0.3,
        vtail_stop: 0.6,
        geometry_unit_m: 1.0,
        template_file: "gf180mcuD.spice",
        output_directory: "shapeic-ota-4t-gf180-pre-layout",
        flavor: PdkFlavor::Gf180,
    }
}

#[derive(Clone, Copy)]
struct DeviceColumns {
    width: &'static str,
    finger_width: &'static str,
    length: &'static str,
    nf: &'static str,
    vbs: &'static str,
    vgs: &'static str,
    vds: &'static str,
}

#[derive(Clone, Copy)]
struct DeviceResult {
    width: f64,
    finger_width: f64,
    length: f64,
    nf: u32,
    vbs: f64,
    vgs: f64,
    vds: f64,
}

#[derive(Clone, Copy)]
struct CandidateResult {
    dp_index: usize,
    cm_index: usize,
    vout: f64,
    vbias: f64,
    n1: f64,
    dp: DeviceResult,
    cm: DeviceResult,
    metrics: [Option<f64>; 4],
}

pub fn run(spec: OtaPreLayoutSpec) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (nmos_path, pmos_path) = lut_paths(spec.label)?;
    let pdk_directory = resolve_pdk(&spec)?;
    let output_root = unique_output_root(manifest, spec.output_directory)?;

    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    validate_lut(&nmos_table, &spec, spec.nmos_model, "NMOS")?;
    validate_lut(&pmos_table, &spec, spec.pmos_model, "PMOS")?;
    let nmos = nmos_table.model(spec.nmos_model)?;
    let pmos = pmos_table.model(spec.pmos_model)?;

    let primitives = load_primitive_catalog(&manifest.join("../analoglib/primitives"))
        .map_err(|error| io::Error::other(format!("could not load primitives: {error:?}")))?;
    let macro_ = ota_macro(manifest.join("examples/ota_4t_pre_layout/gain.spice"));
    let macros = MacroCatalog::from_macros([macro_.clone()])?;
    let input = exploration_input(&spec, nmos, pmos)?;
    let result = macro_.explore(&primitives, &macros, input)?;
    let candidates = collect_candidates(&result)?;
    let validation_candidate = candidates
        .iter()
        .copied()
        .find(|candidate| {
            candidate
                .metrics
                .iter()
                .all(|metric| metric.is_some_and(f64::is_finite))
        })
        .ok_or_else(|| {
            io::Error::other("exploration produced no candidate with all four AC metrics")
        })?;

    fs::create_dir_all(&output_root)?;
    let report = verify_candidate(
        &spec,
        manifest,
        &pdk_directory,
        &output_root,
        validation_candidate,
    )?;
    let csv_path = output_root.join("results.csv");
    write_results_csv(&spec, &candidates, validation_candidate, &report, &csv_path)?;

    print_summary(&spec, &result, &candidates, validation_candidate, &report);
    println!("Results CSV: {}", csv_path.display());
    println!("Artifacts: {}", output_root.display());
    println!("Total time: {:?}", started.elapsed());

    let run = report
        .runs
        .first()
        .ok_or_else(|| io::Error::other("NGSpice verification produced no run"))?;
    if run.status != VerificationStatus::Success {
        return Err(io::Error::other(format!(
            "NGSpice verification did not produce every requested metric: {:?}",
            run.status
        ))
        .into());
    }
    Ok(())
}

fn validate_lut(
    table: &LookupTable,
    spec: &OtaPreLayoutSpec,
    model: &str,
    role: &str,
) -> Result<(), io::Error> {
    let actual_models = table.model_names().collect::<Vec<_>>();
    let metadata_matches = table.pdk() == Some(spec.pdk)
        && table.pdk_revision() == Some(spec.revision)
        && table.corner() == Some(spec.corner)
        && table.temperature_c() == Some(spec.temperature_c)
        && table.nominal_voltage() == Some(spec.nominal_voltage)
        && actual_models == [model];
    if !metadata_matches {
        return Err(io::Error::other(format!(
            "{role} LUT does not match {} {} at {} C: pdk={:?}, revision={:?}, corner={:?}, nominal_voltage={:?}, models={actual_models:?}",
            spec.pdk,
            spec.corner,
            spec.temperature_c,
            table.pdk(),
            table.pdk_revision(),
            table.corner(),
            table.nominal_voltage(),
        )));
    }
    Ok(())
}

fn exploration_input<'a>(
    spec: &OtaPreLayoutSpec,
    nmos: &'a shapeic_lut::DeviceLut,
    pmos: &'a shapeic_lut::DeviceLut,
) -> Result<MacroExplorationInput<'a>, Box<dyn Error>> {
    let vouts = linspace(spec.vout_start, spec.vout_stop, 5);
    let mut input = MacroExplorationInput::new();
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    input.register_primitive_instance(
        DIFF_PAIR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            PrimitiveBuildInput::new(HashMap::from([
                (
                    "current".to_owned(),
                    PrimitiveBuildValue::Scalar(spec.tail_current),
                ),
                ("VINP".to_owned(), PrimitiveBuildValue::Scalar(spec.vin)),
                (
                    "VOUTP".to_owned(),
                    PrimitiveBuildValue::Vector(vouts.clone()),
                ),
                (
                    "VTAIL".to_owned(),
                    PrimitiveBuildValue::Vector(linspace(spec.vtail_start, spec.vtail_stop, 5)),
                ),
            ])),
            vec![CandidateFilter::at_most(DP_COLUMNS.width, MAX_TOTAL_WIDTH)?],
        ),
    )?;
    input.register_primitive_instance(
        CURRENT_MIRROR_INSTANCE,
        PrimitiveInstanceExplorationInput::new(
            PrimitiveBuildInput::new(HashMap::from([
                (
                    "current".to_owned(),
                    PrimitiveBuildValue::Scalar(spec.tail_current),
                ),
                (
                    "VINP".to_owned(),
                    PrimitiveBuildValue::Scalar(spec.mirror_reference),
                ),
                ("VOUTP".to_owned(), PrimitiveBuildValue::Vector(vouts)),
                ("VDD".to_owned(), PrimitiveBuildValue::Scalar(spec.vdd)),
            ])),
            vec![CandidateFilter::at_most(CM_COLUMNS.width, MAX_TOTAL_WIDTH)?],
        ),
    )?;
    Ok(input)
}

fn ota_macro(testbench_path: PathBuf) -> Macro {
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
    let compact = Circuit::builder()
        .vccs("gm_dp_m1", "VOUT", "IBIAS", "VINP", "IBIAS", "gm_dp")
        .resistor("ro_dp_m1", "VOUT", "IBIAS", "ro_dp")
        .vccs("gm_dp_m2", "N1", "IBIAS", "VINN", "IBIAS", "gm_dp")
        .resistor("ro_dp_m2", "N1", "IBIAS", "ro_dp")
        .vccs("gm_cm_m1", "VOUT", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m1", "VOUT", "VDD", "ro_cm")
        .vccs("gm_cm_m2", "N1", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m2", "N1", "VDD", "ro_cm")
        .build();
    Macro::new(
        "ota_4t_pre_layout",
        vec![
            MacroPort::new("VINP", MacroPortRole::Input),
            MacroPort::new("VINN", MacroPortRole::Input),
            MacroPort::new("VOUT", MacroPortRole::Output),
            MacroPort::new("IBIAS", MacroPortRole::Bias),
            MacroPort::new("VDD", MacroPortRole::Supply),
        ],
        circuit,
        compact,
    )
    .with_ac_testbench(MacroAcTestbench::from_spice_file(
        AC_TESTBENCH,
        testbench_path,
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
                mode: AnalysisMode::FullInsight,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::ALL,
            },
        ),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "gm_dp",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_GM),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "ro_dp",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_RO),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "gm_cm",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CM_GM),
    ))
    .with_compact_output(MacroCompactOutputBinding::new(
        "ro_cm",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CM_RO),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VINP",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_VIN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VINN",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_VIN),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VOUT",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_VOUT),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "IBIAS",
        MacroOutputSource::candidate_column(DIFF_PAIR_INSTANCE, DP_VTAIL),
    ))
    .with_interface_binding(MacroInterfaceBinding::new(
        "VDD",
        MacroOutputSource::candidate_column(CURRENT_MIRROR_INSTANCE, CM_VDD),
    ))
}

fn collect_candidates(result: &MacroExplorationResult) -> Result<Vec<CandidateResult>, io::Error> {
    let mut candidates = result
        .accepted()
        .iter()
        .map(|accepted| candidate_result(result, accepted))
        .collect::<Result<Vec<_>, _>>()?;
    candidates.sort_by_key(|candidate| (candidate.dp_index, candidate.cm_index));
    Ok(candidates)
}

fn candidate_result(
    result: &MacroExplorationResult,
    accepted: &MacroAcceptedCandidate,
) -> Result<CandidateResult, io::Error> {
    let dp_index = selected_index(result, accepted, DIFF_PAIR_INSTANCE)?;
    let cm_index = selected_index(result, accepted, CURRENT_MIRROR_INSTANCE)?;
    let vout = selected_value(result, accepted, DIFF_PAIR_INSTANCE, DP_VOUT)?;
    let cm_vout = selected_value(result, accepted, CURRENT_MIRROR_INSTANCE, CM_VOUT)?;
    if vout != cm_vout {
        return Err(io::Error::other(
            "joined candidate contains inconsistent VOUT values",
        ));
    }
    let metrics = result
        .ac_outcome(accepted, AC_TESTBENCH)
        .ok_or_else(|| io::Error::other("accepted candidate has no electrical AC outcome"))?
        .metrics;
    Ok(CandidateResult {
        dp_index,
        cm_index,
        vout,
        vbias: selected_value(result, accepted, DIFF_PAIR_INSTANCE, DP_VTAIL)?,
        n1: selected_value(result, accepted, CURRENT_MIRROR_INSTANCE, CM_VREF)?,
        dp: device_result(result, accepted, DIFF_PAIR_INSTANCE, DP_COLUMNS)?,
        cm: device_result(result, accepted, CURRENT_MIRROR_INSTANCE, CM_COLUMNS)?,
        metrics: [
            metrics.dc_gain_db,
            metrics.bandwidth_3db_hz,
            metrics.unity_gain_hz,
            metrics.phase_margin_deg,
        ],
    })
}

fn device_result(
    result: &MacroExplorationResult,
    accepted: &MacroAcceptedCandidate,
    instance: &str,
    columns: DeviceColumns,
) -> Result<DeviceResult, io::Error> {
    let nf = selected_value(result, accepted, instance, columns.nf)?;
    if nf.fract() != 0.0 || nf < 1.0 || nf > f64::from(u32::MAX) {
        return Err(io::Error::other(format!(
            "invalid NF {nf} for '{instance}'"
        )));
    }
    Ok(DeviceResult {
        width: selected_value(result, accepted, instance, columns.width)?,
        finger_width: selected_value(result, accepted, instance, columns.finger_width)?,
        length: selected_value(result, accepted, instance, columns.length)?,
        nf: nf as u32,
        vbs: selected_value(result, accepted, instance, columns.vbs)?,
        vgs: selected_value(result, accepted, instance, columns.vgs)?,
        vds: selected_value(result, accepted, instance, columns.vds)?,
    })
}

fn selected_index(
    result: &MacroExplorationResult,
    accepted: &MacroAcceptedCandidate,
    instance: &str,
) -> Result<usize, io::Error> {
    result
        .selected_candidate_index(accepted, instance)
        .ok_or_else(|| io::Error::other(format!("candidate does not select '{instance}'")))
}

fn selected_value(
    result: &MacroExplorationResult,
    accepted: &MacroAcceptedCandidate,
    instance: &str,
    column: &str,
) -> Result<f64, io::Error> {
    result
        .selected_candidate(accepted, instance)
        .and_then(|candidate| candidate.get(column))
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other(format!("'{instance}' has no finite '{column}' value")))
}

fn verify_candidate(
    spec: &OtaPreLayoutSpec,
    manifest: &Path,
    pdk_directory: &Path,
    output_root: &Path,
    candidate: CandidateResult,
) -> Result<VerificationReport, Box<dyn Error>> {
    let config = verification_config(spec, manifest, pdk_directory, output_root);
    let metrics = complete_metrics(candidate.metrics)?;
    let mut input = VerificationInput::new(
        OperatingPoint::new(
            candidate.dp.length,
            candidate.dp.vbs,
            candidate.dp.vgs,
            candidate.dp.vds,
        ),
        candidate.dp.width,
        candidate.dp.nf,
    )
    .reference("dc_gain_db", metrics[0])
    .reference("bandwidth_3db_hz", metrics[1])
    .reference("unity_gain_hz", metrics[2])
    .reference("phase_margin_deg", metrics[3]);
    for (name, value) in [
        ("dp_length", spice_geometry(spec, candidate.dp.length)),
        ("dp_width", spice_geometry(spec, candidate.dp.width)),
        ("dp_nf", candidate.dp.nf.to_string()),
        ("cm_length", spice_geometry(spec, candidate.cm.length)),
        ("cm_width", spice_geometry(spec, candidate.cm.width)),
        ("cm_nf", candidate.cm.nf.to_string()),
        ("tail_current", format_float(spec.tail_current)),
        ("vdd_dc", format_float(spec.vdd)),
        ("vin_dc", format_float(spec.vin)),
        ("vout_dc", format_float(candidate.vout)),
    ] {
        input = input.dynamic_template_variable(name, value);
    }
    Ok(VerificationEngine::new(config)?.verify_one(&input)?)
}

fn verification_config(
    spec: &OtaPreLayoutSpec,
    manifest: &Path,
    pdk_directory: &Path,
    output_root: &Path,
) -> VerificationConfig {
    let template = manifest
        .join("examples/ota_4t_pre_layout")
        .join(spec.template_file);
    let mut config = VerificationConfig::new(template, output_root.join("verification"))
        .max_width_per_finger(MAX_FINGER_WIDTH)
        .template_variable("temperature_c", format_float(spec.temperature_c))
        .template_variable(
            "output_bias_inductance",
            format_float(OUTPUT_BIAS_INDUCTANCE_H),
        )
        .template_variable(
            "ac_points_per_decade",
            NGSPICE_POINTS_PER_DECADE.to_string(),
        )
        .template_variable("ac_min_hz", format_float(AC_MIN_HZ))
        .template_variable("ac_max_hz", format_float(AC_MAX_HZ));
    config = match spec.flavor {
        PdkFlavor::Sky130 => config.template_variable(
            "model_library",
            pdk_directory
                .join("libs.tech/combined/sky130.lib.spice")
                .display()
                .to_string(),
        ),
        PdkFlavor::Gf180 => config
            .template_variable(
                "design_include",
                pdk_directory
                    .join("libs.tech/ngspice/design.spice")
                    .display()
                    .to_string(),
            )
            .template_variable(
                "model_library",
                pdk_directory
                    .join("libs.tech/ngspice/sm141064.spice")
                    .display()
                    .to_string(),
            ),
    };
    for variable in VERIFICATION_DYNAMIC_VARIABLES {
        config = config.dynamic_template_variable(variable);
    }
    config
}

fn complete_metrics(metrics: [Option<f64>; 4]) -> Result<[f64; 4], io::Error> {
    let values = metrics.map(|value| value.filter(|value| value.is_finite()));
    let [Some(gain), Some(bandwidth), Some(unity), Some(phase_margin)] = values else {
        return Err(io::Error::other(
            "candidate does not contain all four finite AC metrics",
        ));
    };
    Ok([gain, bandwidth, unity, phase_margin])
}

fn spice_geometry(spec: &OtaPreLayoutSpec, value: f64) -> String {
    format_float(value / spec.geometry_unit_m)
}

fn write_results_csv(
    spec: &OtaPreLayoutSpec,
    candidates: &[CandidateResult],
    validated: CandidateResult,
    report: &VerificationReport,
    path: &Path,
) -> Result<(), io::Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "candidate_id,dp_index,cm_index,validated,vin_v,vdd_v,tail_current_a,vout_v,vbias_v,n1_v,dp_width_m,dp_finger_width_m,dp_length_m,dp_nf,dp_vbs_v,dp_vgs_v,dp_vds_v,cm_width_m,cm_finger_width_m,cm_length_m,cm_nf,cm_vbs_v,cm_vgs_v,cm_vds_v,shapeic_dc_gain_db,shapeic_bandwidth_3db_hz,shapeic_unity_gain_hz,shapeic_phase_margin_deg,ngspice_dc_gain_db,ngspice_bandwidth_3db_hz,ngspice_unity_gain_hz,ngspice_phase_margin_deg,dc_gain_abs_error,dc_gain_error_percent,bandwidth_abs_error_hz,bandwidth_error_percent,unity_gain_abs_error_hz,unity_gain_error_percent,phase_margin_abs_error_deg,phase_margin_error_percent"
    )?;
    for (candidate_id, candidate) in candidates.iter().enumerate() {
        let is_validated =
            candidate.dp_index == validated.dp_index && candidate.cm_index == validated.cm_index;
        write!(
            writer,
            "{candidate_id},{},{},{is_validated},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e}",
            candidate.dp_index,
            candidate.cm_index,
            spec.vin,
            spec.vdd,
            spec.tail_current,
            candidate.vout,
            candidate.vbias,
            candidate.n1,
            candidate.dp.width,
            candidate.dp.finger_width,
            candidate.dp.length,
            candidate.dp.nf,
            candidate.dp.vbs,
            candidate.dp.vgs,
            candidate.dp.vds,
            candidate.cm.width,
            candidate.cm.finger_width,
            candidate.cm.length,
            candidate.cm.nf,
            candidate.cm.vbs,
            candidate.cm.vgs,
            candidate.cm.vds,
        )?;
        for metric in candidate.metrics {
            write_optional(&mut writer, metric)?;
        }
        if is_validated {
            for name in [
                "dc_gain_db",
                "bandwidth_3db_hz",
                "unity_gain_hz",
                "phase_margin_deg",
            ] {
                let row = report
                    .rows
                    .iter()
                    .find(|row| row.metric == name)
                    .ok_or_else(|| io::Error::other(format!("verification has no '{name}' row")))?;
                write_optional(&mut writer, row.simulation_value)?;
            }
            for name in [
                "dc_gain_db",
                "bandwidth_3db_hz",
                "unity_gain_hz",
                "phase_margin_deg",
            ] {
                let row = report
                    .rows
                    .iter()
                    .find(|row| row.metric == name)
                    .expect("verification row checked above");
                write_optional(&mut writer, row.absolute_error)?;
                write_optional(&mut writer, row.percent_error)?;
            }
        } else {
            for _ in 0..12 {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }
    writer.flush()
}

fn write_optional(writer: &mut impl Write, value: Option<f64>) -> Result<(), io::Error> {
    write!(writer, ",")?;
    if let Some(value) = value {
        write!(writer, "{value:.17e}")?;
    }
    Ok(())
}

fn print_summary(
    spec: &OtaPreLayoutSpec,
    result: &MacroExplorationResult,
    candidates: &[CandidateResult],
    validated: CandidateResult,
    report: &VerificationReport,
) {
    println!("{} four-transistor OTA pre-layout exploration", spec.label);
    println!(
        "Compatible candidates: {}; retained candidates: {}",
        result.statistics().compatible_candidates(),
        candidates.len()
    );
    println!(
        "Validated candidate: dp={}, cm={}, VOUT={:.4} V, VBIAS={:.4} V",
        validated.dp_index, validated.cm_index, validated.vout, validated.vbias
    );
    println!(
        "DP: W={:.4} um, Wf={:.4} um, L={:.4} um, NF={}",
        validated.dp.width * 1.0e6,
        validated.dp.finger_width * 1.0e6,
        validated.dp.length * 1.0e6,
        validated.dp.nf
    );
    println!(
        "CM: W={:.4} um, Wf={:.4} um, L={:.4} um, NF={}",
        validated.cm.width * 1.0e6,
        validated.cm.finger_width * 1.0e6,
        validated.cm.length * 1.0e6,
        validated.cm.nf
    );
    println!("\nShapeIC electrical MNA vs. transistor-level NGSpice");
    print!("{}", report.render_table());
}

fn resolve_pdk(spec: &OtaPreLayoutSpec) -> Result<PathBuf, io::Error> {
    let selected = env::var("PDK")
        .map_err(|_| io::Error::other(format!("PDK must be set to '{}'", spec.pdk)))?;
    if selected != spec.pdk {
        return Err(io::Error::other(format!(
            "PDK selects '{selected}', but this example requires '{}'",
            spec.pdk
        )));
    }
    let root = env::var_os("PDK_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("PDK_ROOT must be set"))?;
    if !root.is_dir() {
        return Err(io::Error::other(format!(
            "PDK_ROOT is not a directory: {}",
            root.display()
        )));
    }
    let directory = root.join(&selected);
    if !directory.is_dir() {
        return Err(io::Error::other(format!(
            "PDK '{}' was not found at {}",
            spec.pdk,
            directory.display()
        )));
    }
    Ok(directory)
}

fn lut_paths(label: &str) -> Result<(PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_pre_layout"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz>  # {label}",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into()))
}

fn unique_output_root(manifest: &Path, directory: &str) -> Result<PathBuf, io::Error> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_millis();
    Ok(manifest
        .join("../target")
        .join(directory)
        .join(timestamp.to_string()))
}

fn format_float(value: f64) -> String {
    format!("{value:.17e}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use shapeic_core::macro_model::{MacroAnalysisDomain, validate_macro};
    use shapeic_lut::verification::{MetricStatus, VerificationRow};

    #[test]
    fn pdk_specs_use_conservative_approved_biases() {
        let sky = sky130_spec();
        assert_eq!((sky.vdd, sky.vin, sky.mirror_reference), (1.8, 0.9, 0.6));
        assert_eq!((sky.vout_start, sky.vout_stop), (0.8, 1.2));
        assert_eq!((sky.vtail_start, sky.vtail_stop), (0.1, 0.3));
        let gf = gf180_spec();
        assert_eq!((gf.vdd, gf.vin, gf.mirror_reference), (3.3, 1.4, 2.1));
        assert_eq!((gf.vout_start, gf.vout_stop), (1.6, 2.4));
        assert_eq!((gf.vtail_start, gf.vtail_stop), (0.3, 0.6));
    }

    #[test]
    fn ota_contains_only_one_electrical_testbench() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let primitives = load_primitive_catalog(&manifest.join("../analoglib/primitives")).unwrap();
        let macro_ = ota_macro(manifest.join("examples/ota_4t_pre_layout/gain.spice"));
        let macros = MacroCatalog::from_macros([macro_.clone()]).unwrap();
        let [testbench] = macro_.exploration().testbenches() else {
            panic!("pre-layout OTA must contain exactly one testbench");
        };
        assert_eq!(testbench.name(), AC_TESTBENCH);
        assert_eq!(testbench.domain(), MacroAnalysisDomain::Electrical);
        assert_eq!(testbench.analysis().policy, AdaptiveAcPolicy::COMPLETE);
        assert!(validate_macro(&macro_, &primitives, &macros).is_empty());
    }

    #[test]
    fn both_ngspice_templates_have_complete_variable_contracts() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        for spec in [sky130_spec(), gf180_spec()] {
            let template = manifest
                .join("examples/ota_4t_pre_layout")
                .join(spec.template_file);
            let source = fs::read_to_string(&template).unwrap();
            assert!(source.contains("set wr_vecnames"));
            assert!(source.contains("set wr_singlescale"));
            let config = verification_config(
                &spec,
                manifest,
                Path::new("/installed/pdk"),
                Path::new("/temporary/output"),
            );
            VerificationEngine::new(config).expect("verification template must be complete");
        }
    }

    #[test]
    fn results_csv_keeps_a_stable_column_count() {
        let device = DeviceResult {
            width: 10.0e-6,
            finger_width: 5.0e-6,
            length: 0.8e-6,
            nf: 2,
            vbs: 0.0,
            vgs: 0.7,
            vds: 0.8,
        };
        let validated = CandidateResult {
            dp_index: 0,
            cm_index: 1,
            vout: 1.0,
            vbias: 0.2,
            n1: 0.6,
            dp: device,
            cm: device,
            metrics: [Some(20.0), Some(1.0e6), Some(1.0e7), Some(60.0)],
        };
        let other = CandidateResult {
            dp_index: 2,
            cm_index: 3,
            ..validated
        };
        let point = OperatingPoint::new(device.length, device.vbs, device.vgs, device.vds);
        let names = [
            "dc_gain_db",
            "bandwidth_3db_hz",
            "unity_gain_hz",
            "phase_margin_deg",
        ];
        let report = VerificationReport {
            output_dir: PathBuf::from("output"),
            summary_path: PathBuf::from("output/summary.csv"),
            runs: Vec::new(),
            rows: names
                .iter()
                .enumerate()
                .map(|(index, name)| VerificationRow {
                    point_index: 0,
                    operating_point: point,
                    width: device.width,
                    nf: device.nf,
                    metric: (*name).to_owned(),
                    status: MetricStatus::Compared,
                    reference_value: validated.metrics[index].unwrap(),
                    simulation_value: Some(validated.metrics[index].unwrap() * 1.01),
                    absolute_error: Some(1.0),
                    percent_error: Some(1.0),
                    message: None,
                })
                .collect(),
        };
        let path = env::temp_dir().join(format!(
            "shapeic-ota-pre-layout-csv-{}-{}.csv",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_results_csv(
            &sky130_spec(),
            &[validated, other],
            validated,
            &report,
            &path,
        )
        .unwrap();
        let csv = fs::read_to_string(&path).unwrap();
        let counts = csv
            .lines()
            .map(|line| line.split(',').count())
            .collect::<Vec<_>>();
        assert_eq!(counts, vec![40, 40, 40]);
        fs::remove_file(path).unwrap();
    }
}
