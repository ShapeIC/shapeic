use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

use shapeic_core::analysis::{
    AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::exploration::filter::CandidateFilter;
use shapeic_core::macro_model::{
    Macro, MacroAcTestbench, MacroCatalog, MacroCompactOutputBinding, MacroCompactSeedSet,
    MacroDerivationReduction, MacroDerivationRule, MacroDerivationTarget, MacroDesignVariable,
    MacroDesignVariableBinding, MacroExecutionConfig, MacroHierarchyExplorationInput,
    MacroHierarchyExplorationResult, MacroHierarchyPath, MacroHierarchyPathInput,
    MacroInterfaceBinding, MacroOutputSource, MacroPort, MacroPortRole, MacroPrimitiveDefault,
    MacroPublicInputAlias, MacroSpecification, MacroSpecificationBounds, MacroSpecificationSource,
    explore_macro_hierarchy,
};
use shapeic_core::primitive::build::{
    PrimitiveBuildInput, PrimitiveBuildInputKind, PrimitiveBuildValue,
};
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::utils::linspace;
use shapeic_lut::LookupTable;

const TOP_MACRO: &str = "ota_system";
const OTA_MACRO: &str = "ota_4t";
const OTA_INSTANCE: &str = "xota";
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";
const SYSTEM_TESTBENCH: &str = "system_gain";
const OTA_TESTBENCH: &str = "ota_gain";
const GAIN_SPECIFICATION: &str = "dc_gain_db";

const VOUT_POINTS: usize = 100;
const VBIAS_POINTS: usize = 100;
const MAX_WIDTH: f64 = 100.0e-6;
const MAX_PRINTED_SOLUTIONS: usize = 20;

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

const WIDTH_COLUMN_SUFFIX: &str = "width";
const FINGER_WIDTH_COLUMN_SUFFIX: &str = "finger_width";
const LENGTH_COLUMN_SUFFIX: &str = "length";
const NF_COLUMN_SUFFIX: &str = "nf";
const VBS_COLUMN_SUFFIX: &str = "vbs";
const VGS_COLUMN_SUFFIX: &str = "vgs";
const VDS_COLUMN_SUFFIX: &str = "vds";

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
};

fn main() -> Result<(), Box<dyn Error>> {
    let options = cli_options()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let testbench = manifest.join("examples/ota_4t_hierarchical/gain.spice");
    let primitive_catalog = load_primitive_catalog(&manifest.join("../shapeic-cellkit/primitives"))
        .map_err(|error| format!("{error:?}"))?;

    let nmos_table = LookupTable::open(&options.nmos_path)?;
    let pmos_table = LookupTable::open(&options.pmos_path)?;
    let spec = resolve_pdk_spec(&nmos_table, &pmos_table)?;
    let nmos = nmos_table.model(spec.nmos_model)?;
    let pmos = pmos_table.model(spec.pmos_model)?;

    let ota = ota_macro(spec, testbench.clone());
    let system = system_macro(testbench);
    let macro_catalog = MacroCatalog::from_macros([ota, system])?;
    let mut input = MacroHierarchyExplorationInput::new();
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    input.register_path(MacroHierarchyPath::root(TOP_MACRO)?, top_path_input(spec))?;
    input.set_execution_config(options.execution_config()?);

    let result = explore_macro_hierarchy(TOP_MACRO, &macro_catalog, &primitive_catalog, &input)?;
    validate_result(&result)?;

    println!("{} hierarchical four-transistor OTA", spec.label);
    print_hierarchy(&result);
    print_derivations(&result);
    print_solutions(&result)?;
    Ok(())
}

fn ota_macro(spec: PdkSpec, testbench: PathBuf) -> Macro {
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

    Macro::new(OTA_MACRO, ota_ports(), circuit, ota_compact_model())
        .with_ac_testbench(MacroAcTestbench::from_spice_file(
            OTA_TESTBENCH,
            testbench,
            ac_analysis(),
        ))
        .with_specification(MacroSpecification::new(
            GAIN_SPECIFICATION,
            MacroSpecificationSource::ac_metric(OTA_TESTBENCH, AcMetric::DcGainDb),
            MacroSpecificationBounds::unbounded(),
        ))
        .with_primitive_default(MacroPrimitiveDefault::new(
            DIFF_PAIR_INSTANCE,
            diff_pair_input(spec),
            vec![
                CandidateFilter::at_most(DIFF_PAIR_WIDTH_COLUMN, MAX_WIDTH)
                    .expect("finite width filter"),
            ],
        ))
        .with_primitive_default(MacroPrimitiveDefault::new(
            CURRENT_MIRROR_INSTANCE,
            current_mirror_input(spec),
            vec![
                CandidateFilter::at_most(CURRENT_MIRROR_WIDTH_COLUMN, MAX_WIDTH)
                    .expect("finite width filter"),
            ],
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vout",
            PrimitiveBuildInputKind::Vector,
            vec![
                MacroDesignVariableBinding::new(DIFF_PAIR_INSTANCE, "VOUTP"),
                MacroDesignVariableBinding::new(CURRENT_MIRROR_INSTANCE, "VOUTP"),
            ],
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vbias",
            PrimitiveBuildInputKind::Vector,
            vec![MacroDesignVariableBinding::new(DIFF_PAIR_INSTANCE, "VTAIL")],
        ))
        .with_compact_seeds(ota_seeds())
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

fn system_macro(testbench: PathBuf) -> Macro {
    let circuit = Circuit::builder()
        .macro_instance(
            OTA_INSTANCE,
            OTA_MACRO,
            [
                ("VINP", "VINP"),
                ("VINN", "VINN"),
                ("VOUT", "VOUT"),
                ("IBIAS", "IBIAS"),
                ("VDD", "VDD"),
            ],
        )
        .build();

    Macro::new(TOP_MACRO, ota_ports(), circuit, ota_compact_model())
        .with_ac_testbench(MacroAcTestbench::from_spice_file(
            SYSTEM_TESTBENCH,
            testbench,
            ac_analysis(),
        ))
        .with_specification(MacroSpecification::new(
            GAIN_SPECIFICATION,
            MacroSpecificationSource::ac_metric(SYSTEM_TESTBENCH, AcMetric::DcGainDb),
            MacroSpecificationBounds::at_least(MIN_DC_GAIN_DB),
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vout",
            PrimitiveBuildInputKind::Vector,
            Vec::new(),
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vbias",
            PrimitiveBuildInputKind::Vector,
            Vec::new(),
        ))
        .with_public_input_alias(MacroPublicInputAlias::new(
            "vout",
            OTA_INSTANCE,
            "vout",
            "VOUT",
        ))
        .with_public_input_alias(MacroPublicInputAlias::new(
            "vbias",
            OTA_INSTANCE,
            "vbias",
            "IBIAS",
        ))
        .with_derivation_rule(MacroDerivationRule::new(
            OTA_INSTANCE,
            MIN_DC_GAIN_DB.to_string(),
            MacroDerivationReduction::Minimum,
            MacroDerivationTarget::specification_minimum(GAIN_SPECIFICATION),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_dp",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "gm_dp__xota"),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "ro_dp",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "ro_dp__xota"),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_cm",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "gm_cm__xota"),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "ro_cm",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "ro_cm__xota"),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VINP",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "xota.vinp"),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VINN",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "xota.vinn"),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VOUT",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "xota.vout"),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "IBIAS",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "xota.ibias"),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VDD",
            MacroOutputSource::candidate_column(OTA_INSTANCE, "xota.vdd"),
        ))
}

fn top_path_input(spec: PdkSpec) -> MacroHierarchyPathInput {
    let mut input = MacroHierarchyPathInput::new();
    input
        .register_design_variable_override(
            "vout",
            PrimitiveBuildValue::Vector(linspace(spec.vout_start, spec.vout_stop, VOUT_POINTS)),
        )
        .expect("top-level VOUT is registered once");
    input
        .register_design_variable_override(
            "vbias",
            PrimitiveBuildValue::Vector(linspace(spec.vbias_start, spec.vbias_stop, VBIAS_POINTS)),
        )
        .expect("top-level VBIAS is registered once");
    input
}

fn ota_ports() -> Vec<MacroPort> {
    vec![
        MacroPort::new("VINP", MacroPortRole::Input),
        MacroPort::new("VINN", MacroPortRole::Input),
        MacroPort::new("VOUT", MacroPortRole::Output),
        MacroPort::new("IBIAS", MacroPortRole::Bias),
        MacroPort::new("VDD", MacroPortRole::Supply),
    ]
}

fn ota_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_dp_m1", "VOUT", "IBIAS", "VINP", "IBIAS", "gm_dp")
        .resistor("ro_dp_m1", "VOUT", "IBIAS", "ro_dp")
        .vccs("gm_dp_m2", "N1", "IBIAS", "VINN", "IBIAS", "gm_dp")
        .resistor("ro_dp_m2", "N1", "IBIAS", "ro_dp")
        .vccs("gm_cm_m1", "VOUT", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m1", "VOUT", "VDD", "ro_cm")
        .vccs("gm_cm_m2", "N1", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m2", "N1", "VDD", "ro_cm")
        .capacitor("c_out", "VOUT", "IBIAS", 1.0e-12)
        .build()
}

fn ota_seeds() -> MacroCompactSeedSet {
    MacroCompactSeedSet::aligned([
        ("gm_dp", vec![0.7e-3, 1.0e-3, 1.3e-3]),
        ("ro_dp", vec![140.0e3, 100.0e3, 75.0e3]),
        ("gm_cm", vec![0.7e-3, 1.0e-3, 1.3e-3]),
        ("ro_cm", vec![140.0e3, 100.0e3, 75.0e3]),
    ])
}

fn ac_analysis() -> AcAnalysis {
    AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT").with_polarity(TransferPolarity::Negative),
        AdaptiveAcConfig {
            min_frequency_hz: 1.0,
            max_frequency_hz: 100.0e9,
            coarse_points_per_decade: 4,
            crossing_relative_tolerance: 0.005,
            max_refinement_steps: 32,
            retain_samples: false,
        },
        AdaptiveAcPolicy {
            mode: AnalysisMode::Prune,
            targets: AnalysisTargets {
                min_dc_gain_db: None,
                min_bandwidth_3db_hz: Some(MIN_BANDWIDTH_3DB_HZ),
                min_unity_gain_hz: Some(MIN_UNITY_GAIN_HZ),
                min_phase_margin_deg: Some(MIN_PHASE_MARGIN_DEG),
            },
            metrics: AcMetricSet::ALL,
        },
    )
}

fn diff_pair_input(spec: PdkSpec) -> PrimitiveBuildInput {
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

fn current_mirror_input(spec: PdkSpec) -> PrimitiveBuildInput {
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

fn validate_result(result: &MacroHierarchyExplorationResult) -> Result<(), io::Error> {
    if result.root_result().accepted().is_empty() {
        return Err(io::Error::other(
            "hierarchical exploration produced no accepted parent candidates",
        ));
    }
    let expected_child = result
        .root_path()
        .child(OTA_INSTANCE)
        .map_err(io::Error::other)?;
    let derivation = result
        .derivation(&expected_child)
        .ok_or_else(|| io::Error::other("the OTA child has no derivation record"))?;
    if derivation.conditions().audit().len() != 1
        || derivation.conditions().public_input_audit().len() != 2
        || !derivation
            .conditions()
            .specification_bounds()
            .contains_key(GAIN_SPECIFICATION)
        || !derivation
            .conditions()
            .design_variable_conditions()
            .contains_key("vout")
        || !derivation
            .conditions()
            .design_variable_conditions()
            .contains_key("vbias")
    {
        return Err(io::Error::other(
            "the OTA child did not receive every expected derived condition",
        ));
    }
    for accepted_index in 0..result.root_result().accepted().len() {
        let selection = result.selection(accepted_index).map_err(io::Error::other)?;
        if selection.nodes().count() != 2 || selection.primitive_instances().count() != 2 {
            return Err(io::Error::other(format!(
                "accepted root row {accepted_index} does not resolve one OTA and two primitives"
            )));
        }
    }
    Ok(())
}

fn print_hierarchy(result: &MacroHierarchyExplorationResult) {
    println!("\nHierarchy execution");
    for (path, node) in result.nodes() {
        let accepted = node.result().map_or(0, |result| result.accepted().len());
        println!(
            "  {path}: macro={}, status={:?}, accepted={accepted}",
            node.macro_name(),
            node.status()
        );
    }
    let statistics = result.statistics();
    println!(
        "  paths={}, previews={}, definitive={}, frequency_evaluations={}, total={:?}",
        statistics.total_paths(),
        statistics.previews_executed(),
        statistics.definitive_evaluations(),
        statistics.frequency_evaluations(),
        statistics.total_duration(),
    );
}

fn print_derivations(result: &MacroHierarchyExplorationResult) {
    println!("\nParent-to-child derivations");
    for (_, derivation) in result.derivations() {
        println!(
            "  {} -> {}",
            derivation.parent_path(),
            derivation.child_path()
        );
        for entry in derivation.conditions().audit() {
            println!(
                "    {}: {:?} over {} row(s) => {:?}; {:?}",
                entry.expression(),
                entry.reduction(),
                entry.source_rows(),
                entry.reduced_value(),
                entry.effective_condition(),
            );
        }
        for entry in derivation.conditions().public_input_audit() {
            println!(
                "    input {} via {}.{} -> {}: {:?} over {} row(s); {:?}",
                entry.parent_variable(),
                entry.child_instance(),
                entry.interface_port(),
                entry.child_variable(),
                entry.values(),
                entry.source_rows(),
                entry.effective_condition(),
            );
        }
    }
}

fn print_solutions(result: &MacroHierarchyExplorationResult) -> Result<(), io::Error> {
    println!("\nAccepted hierarchical solutions");
    println!(
        "{:<6} {:<21} {:>8} {:>8} {:>9} {:>9} {:>8} {:>5} {:>8} {:>8} {:>8}",
        "root", "instance", "index", "vout", "w_um", "wf_um", "l_um", "nf", "vbs", "vgs", "vds"
    );
    for accepted_index in 0..result
        .root_result()
        .accepted()
        .len()
        .min(MAX_PRINTED_SOLUTIONS)
    {
        let selection = result.selection(accepted_index).map_err(io::Error::other)?;
        let root = selection
            .node(result.root_path())
            .ok_or_else(|| io::Error::other("selection has no root node"))?;
        let child_path = result
            .root_path()
            .child(OTA_INSTANCE)
            .map_err(io::Error::other)?;
        let child = selection
            .node(&child_path)
            .ok_or_else(|| io::Error::other("selection has no OTA child node"))?;
        print_metrics("system", root.ac_outcome(SYSTEM_TESTBENCH))?;
        print_metrics("ota", child.ac_outcome(OTA_TESTBENCH))?;

        for instance in selection.primitive_instances() {
            let local = instance.instance();
            println!(
                "{:<6} {:<21} {:>8} {:>8.4} {:>9.4} {:>9.4} {:>8.4} {:>5.0} {:>8.4} {:>8.4} {:>8.4}",
                accepted_index,
                instance.path(),
                instance.candidate_index(),
                required_value(&instance, interface_column(local, "voutp"))?,
                required_value(&instance, mos_column(WIDTH_COLUMN_SUFFIX, local))? * 1.0e6,
                required_value(&instance, mos_column(FINGER_WIDTH_COLUMN_SUFFIX, local))? * 1.0e6,
                required_value(&instance, mos_column(LENGTH_COLUMN_SUFFIX, local))? * 1.0e6,
                required_value(&instance, mos_column(NF_COLUMN_SUFFIX, local))?,
                required_value(&instance, mos_column(VBS_COLUMN_SUFFIX, local))?,
                required_value(&instance, mos_column(VGS_COLUMN_SUFFIX, local))?,
                required_value(&instance, mos_column(VDS_COLUMN_SUFFIX, local))?,
            );
        }
    }
    let accepted = result.root_result().accepted().len();
    if accepted > MAX_PRINTED_SOLUTIONS {
        println!(
            "  ... {} additional accepted solutions were validated but not printed",
            accepted - MAX_PRINTED_SOLUTIONS
        );
    }
    Ok(())
}

fn print_metrics(
    label: &str,
    outcome: Option<&shapeic_core::analysis::AdaptiveAcOutcome>,
) -> Result<(), io::Error> {
    let metrics = outcome
        .map(|outcome| &outcome.metrics)
        .ok_or_else(|| io::Error::other(format!("selected {label} result has no AC outcome")))?;
    println!(
        "  {label}: gain={:.4} dB, f3dB={:.6e} Hz, UGF={:.6e} Hz, PM={:.4} deg",
        metric(metrics.dc_gain_db, "DC gain")?,
        metric(metrics.bandwidth_3db_hz, "bandwidth")?,
        metric(metrics.unity_gain_hz, "UGF")?,
        metric(metrics.phase_margin_deg, "phase margin")?,
    );
    Ok(())
}

fn required_value(
    instance: &shapeic_core::macro_model::MacroHierarchySelectedInstance<'_>,
    column: String,
) -> Result<f64, io::Error> {
    instance
        .value(&column)
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other(format!("{} has no finite '{column}'", instance.path())))
}

fn metric(value: Option<f64>, name: &str) -> Result<f64, io::Error> {
    value
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other(format!("AC outcome has no finite {name}")))
}

fn mos_column(parameter: &str, instance: &str) -> String {
    format!("{parameter}__{instance}__m1")
}

fn interface_column(instance: &str, port: &str) -> String {
    format!("{instance}.{port}")
}

const DIFF_PAIR_WIDTH_COLUMN: &str = "width__xdp__m1";
const CURRENT_MIRROR_WIDTH_COLUMN: &str = "width__xcm__m1";

fn resolve_pdk_spec(nmos: &LookupTable, pmos: &LookupTable) -> Result<PdkSpec, io::Error> {
    let nmos_spec = identify_pdk(nmos, true)?;
    let pmos_spec = identify_pdk(pmos, false)?;
    if nmos_spec != pmos_spec {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "NMOS LUT uses PDK '{}', but PMOS LUT uses '{}'",
                nmos_spec.pdk, pmos_spec.pdk
            ),
        ));
    }
    Ok(nmos_spec)
}

fn identify_pdk(table: &LookupTable, nmos: bool) -> Result<PdkSpec, io::Error> {
    let candidates = [IHP_SPEC, SKY130_SPEC, GF180_SPEC];
    if let Some(pdk) = table.pdk() {
        return candidates
            .into_iter()
            .find(|spec| spec.pdk == pdk)
            .ok_or_else(|| unsupported_pdk(pdk));
    }
    let names = table.model_names().collect::<Vec<_>>();
    candidates
        .into_iter()
        .find(|spec| {
            names.contains(&if nmos {
                spec.nmos_model
            } else {
                spec.pmos_model
            })
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "LUT metadata and model names do not identify a supported PDK",
            )
        })
}

fn unsupported_pdk(pdk: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported LUT PDK '{pdk}'"),
    )
}

#[derive(Debug)]
struct CliOptions {
    nmos_path: PathBuf,
    pmos_path: PathBuf,
    workers: Option<usize>,
    batch_size: Option<usize>,
}

impl CliOptions {
    fn execution_config(&self) -> Result<MacroExecutionConfig, io::Error> {
        let mut execution = MacroExecutionConfig::sequential();
        if let Some(workers) = self.workers {
            execution = execution
                .with_parallel_candidate_build(workers)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
                .with_parallel_electrical_analysis(workers)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        }
        if let Some(batch_size) = self.batch_size {
            execution = execution
                .with_electrical_batch_size(batch_size)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        }
        Ok(execution)
    }
}

fn cli_options() -> Result<CliOptions, io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_hierarchical"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz> [--workers N] [--batch-size N]",
                Path::new(&executable).display()
            ),
        )
    };
    let mut paths = Vec::new();
    let mut workers = None;
    let mut batch_size = None;
    while let Some(argument) = arguments.next() {
        if argument == "--workers" {
            workers = Some(parse_usize(
                arguments.next().ok_or_else(&usage)?,
                "--workers",
            )?);
        } else if argument == "--batch-size" {
            batch_size = Some(parse_usize(
                arguments.next().ok_or_else(&usage)?,
                "--batch-size",
            )?);
        } else if argument.to_string_lossy().starts_with("--") {
            return Err(usage());
        } else {
            paths.push(PathBuf::from(argument));
        }
    }
    if paths.len() != 2 {
        return Err(usage());
    }
    let options = CliOptions {
        nmos_path: paths.remove(0),
        pmos_path: paths.remove(0),
        workers,
        batch_size,
    };
    options.execution_config()?;
    Ok(options)
}

fn parse_usize(value: OsString, flag: &str) -> Result<usize, io::Error> {
    value
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "argument is not UTF-8"))?
        .parse::<usize>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{flag} requires a positive integer"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shapeic_core::macro_model::{validate_macro_catalog, validate_macro_compact_seeds};

    #[test]
    fn defines_a_valid_parent_child_catalog_and_seed() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let primitives =
            load_primitive_catalog(&manifest.join("../shapeic-cellkit/primitives")).unwrap();
        let testbench = manifest.join("examples/ota_4t_hierarchical/gain.spice");
        let ota = ota_macro(IHP_SPEC, testbench.clone());
        assert!(validate_macro_compact_seeds(&ota).is_empty());
        let catalog = MacroCatalog::from_macros([ota, system_macro(testbench)]).unwrap();
        let errors = validate_macro_catalog(&catalog, &primitives);
        assert!(errors.is_empty(), "invalid hierarchy: {errors:?}");
    }

    #[test]
    fn configures_parallel_execution_consistently() {
        let options = CliOptions {
            nmos_path: "nmos.npz".into(),
            pmos_path: "pmos.npz".into(),
            workers: Some(4),
            batch_size: Some(32),
        };
        let execution = options.execution_config().unwrap();
        assert_eq!(
            execution.electrical_analysis(),
            shapeic_core::macro_model::ElectricalAnalysisExecution::Parallel {
                workers: 4,
                batch_size: 32,
            }
        );
    }

    #[test]
    fn compact_seed_vectors_are_aligned_and_exclude_interface_voltages() {
        let seeds = ota_seeds();
        assert_eq!(seeds.len(), 3);
        assert_eq!(seeds.compact_parameters().len(), 4);
        assert!(
            seeds
                .compact_parameters()
                .iter()
                .all(|(_, values)| values.len() == seeds.len())
        );
        assert!(
            seeds.compact_parameters().iter().all(|(name, _)| !matches!(
                name.as_str(),
                "VINP" | "VINN" | "VOUT" | "IBIAS" | "VDD"
            ))
        );
    }

    #[test]
    fn parent_owns_voltage_grids_and_uses_automatic_child_aliases() {
        let system = system_macro("gain.spice".into());
        let aliases = system.exploration().public_input_aliases();
        assert_eq!(aliases.len(), 2);
        assert_eq!(system.exploration().derivation_rules().len(), 1);
        assert!(aliases.iter().any(|alias| {
            alias.variable() == "vout"
                && alias.child_variable() == "vout"
                && alias.interface_port() == "VOUT"
        }));
        assert!(aliases.iter().any(|alias| {
            alias.variable() == "vbias"
                && alias.child_variable() == "vbias"
                && alias.interface_port() == "IBIAS"
        }));

        let input = top_path_input(GF180_SPEC);
        assert_eq!(
            input.design_variable_override("vout"),
            Some(&PrimitiveBuildValue::Vector(linspace(
                GF180_SPEC.vout_start,
                GF180_SPEC.vout_stop,
                VOUT_POINTS,
            )))
        );
        assert_eq!(
            input.design_variable_override("vbias"),
            Some(&PrimitiveBuildValue::Vector(linspace(
                GF180_SPEC.vbias_start,
                GF180_SPEC.vbias_stop,
                VBIAS_POINTS,
            )))
        );
    }
}
