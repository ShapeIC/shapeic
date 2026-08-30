use std::error::Error;
use shapeic_core::macro_model::MacroExploration;
use shapeic_core::macro_model::explore_macro_hierarchy;
use std::collections::HashMap;
use shapeic_core::macro_model::MacroAcTestbench;
use shapeic_core::macro_model::MacroDesignVariable;
use shapeic_core::macro_model::MacroHierarchyExplorationInput;
use shapeic_core::macro_model::MacroPrimitiveDefault;
use shapeic_core::macro_model::MacroPublicInputAlias;
use shapeic_core::macro_model::MacroSpecification;
use std::ffi::OsString;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::macro_model::{MacroExecutionConfig, Macro, MacroPort, MacroPortRole, MacroCompactSeedSet, MacroSpecificationBounds, MacroSpecificationSource, MacroCatalog, MacroHierarchyExplorationResult, MacroHierarchyMode, MacroHierarchyPathInput, MacroHierarchyPath, MacroHierarchyRetentionPolicy, MacroExplorationStatistics, MacroExecutionReport, MacroHierarchyNodeStatus, MacroDesignVariableBinding};
use shapeic_core::primitive::build::{PrimitiveBuildInputKind, PrimitiveBuildInput, PrimitiveBuildValue};

use shapeic_lut::LookupTable;
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::analysis::{AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets, AcMetricSet, AcMetric};
use shapeic_core::utils::linspace;
use shapeic_core::exploration::filter::CandidateFilter;

const COMMON_SOURCE_INSTANCE: &str = "xcs";
const OTA_1STAGE_INSTANCE: &str = "xota_1stage";
const OTA_1STAGE_TB: &str = "ota_1stage_gain";
const OTA_1STAGE: &str = "ota_1stage"; 
const OTA_2STAGE: &str = "ota_2stage";
const OTA_2STAGE_TB: &str = "ota_2stage_gain";
const DIFF_PAIR_INSTANCE: &str = "xdp";
const CURRENT_MIRROR_INSTANCE: &str = "xcm";


const GAIN_SPECIFICATION: &str = "dc_gain_db";

const VOUT_POINTS: usize = 5;
const VBIAS_POINTS: usize = 5;
const VOUT_1STAGE_POINTS: usize = 3;

const MIN_DC_GAIN_DB: f64 = 40.0;
const MIN_BANDWIDTH_3DB_HZ: f64 = 1.0e6;
const MIN_UNITY_GAIN_HZ: f64 = 1.0e7;
const MIN_PHASE_MARGIN_DEG: f64 = 60.0;

const MAX_WIDTH: f64 = 100.0e-6;


const DIFF_PAIR_WIDTH_COLUMN: &str = "width__xdp__m1";
const CURRENT_MIRROR_WIDTH_COLUMN: &str = "width__xcm__m1";

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
    vout_1stage_start: f64,
    vout_1stage_stop: f64,
    vbias_start: f64,
    vbias_stop: f64,
}

const IHP_SPEC: PdkSpec = PdkSpec {
    label: "IHP SG13G2",
    pdk: "ihp-sg13g2",
    nmos_model: "sg13_lv_nmos",
    pmos_model: "sg13_lv_pmos",
    tail_current: 20.0e-6,
    mirror_reference: 0.9,
    vout_start: 0.95,
    vout_stop: 1.1,
    vdd: 1.5,
    vin: 0.9,
    vout_1stage_start: 0.95,
    vout_1stage_stop: 1.1,
    vbias_start: 0.65,
    vbias_stop: 0.79,
};

fn main() -> Result<(), Box<dyn Error>> {
    let options = cli_options()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gain_2stage_testbench = manifest.join("examples/ota_2stage/gain_2stage.spice");
    let gain_1stage_testbench = manifest.join("examples/ota_2stage/gain_1stage.spice");
    let primitive_catalog = load_primitive_catalog(&manifest.join("../shapeic-cellkit/primitives"))
        .map_err(|error| format!("{error:?}"))?;

    let nmos_table = LookupTable::open(&options.nmos_path)?;
    let pmos_table = LookupTable::open(&options.pmos_path)?;
    let spec = resolve_pdk_spec(&nmos_table, &pmos_table)?;
    let nmos = nmos_table.model(spec.nmos_model)?;
    let pmos = pmos_table.model(spec.pmos_model)?;

    let ota_1stage = ota_1stage(spec, gain_1stage_testbench.clone());
    let ota_2stage = ota_2stage(spec, gain_2stage_testbench.clone());
    let macro_catalog = MacroCatalog::from_macros([ota_1stage, ota_2stage])?;

    let mut input = MacroHierarchyExplorationInput::new();
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    input.register_path(MacroHierarchyPath::root(OTA_2STAGE)?, top_path_input(spec))?;
    input.set_retention_policy(
        MacroHierarchyRetentionPolicy::FullPreviews,
    );
    input.set_execution_config(options.execution_config()?);

    let result = explore_macro_hierarchy(OTA_2STAGE, &macro_catalog, &primitive_catalog, &input)?;

    println!("{} hierarchical four-transistor OTA", spec.label);
    print_hierarchy(&result);
    print_derivations(&result);
    print_results(&result)?;

    Ok(())
}

fn ota_1stage(spec: PdkSpec, testbench: PathBuf) -> Macro {
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
            [
                ("VINP", "N1"), 
                ("VOUTP", "VOUT"), 
                ("VDD", "VDD")],
        )
        .build();
    Macro::new(
        OTA_1STAGE,
        ota_1stage_ports(),
        Circuit::default(),
        ota_1stage_compact_model()
    )
    .with_ac_testbench(MacroAcTestbench::from_spice_file(
        OTA_1STAGE_TB,
        testbench,
        ac_analysis(),
    ))
    .with_specification(MacroSpecification::new(
        GAIN_SPECIFICATION,
        MacroSpecificationSource::ac_metric(OTA_1STAGE_TB, AcMetric::DcGainDb),
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
    .with_compact_seeds(ota_1stage_seed(spec))
    .with_design_variable(MacroDesignVariable::new(
        "vout_1stage",
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
}

fn ota_2stage(spec: PdkSpec, testbench: PathBuf) -> Macro {
    let circuit = Circuit::builder()
        .primitive(
            COMMON_SOURCE_INSTANCE,
            "simplecommonsource",
            [
                ("VIN", "VOUT_1STAGE"),
                ("VOUT", "VOUT"),
                ("VDD", "VDD")
            ]
        )
        .macro_instance(
            OTA_1STAGE_INSTANCE,
            OTA_1STAGE,
            [
                ("VINP", "VINP"),
                ("VINN", "VINN"),
                ("VOUT", "VOUT_1STAGE"),
                ("IBIAS", "IBIAS"),
                ("VDD", "VDD"),
                ("VSS", "VSS")
            ],
        )
        .build();
    Macro::new(OTA_2STAGE, Vec::new(), circuit, ota_2stage_compact_model())
        .with_ac_testbench(MacroAcTestbench::from_spice_file(
            OTA_2STAGE_TB,
            testbench,
            ac_analysis()
        ))
        .with_specification(MacroSpecification::new(
            GAIN_SPECIFICATION,
            MacroSpecificationSource::ac_metric(OTA_2STAGE_TB, AcMetric::DcGainDb),
            MacroSpecificationBounds::at_least(MIN_DC_GAIN_DB),
        ))
        .with_primitive_default(MacroPrimitiveDefault::new(
            COMMON_SOURCE_INSTANCE, 
            common_source_input(spec),
            Vec::new()
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vout_1stage",
            PrimitiveBuildInputKind::Vector,
            Vec::new(),
        ))
        .with_public_input_alias(MacroPublicInputAlias::new(
            "vout_1stage", 
            OTA_1STAGE_INSTANCE, 
            "vout_1stage", 
            "VOUT"
        ))
        .with_specification(MacroSpecification::new(
            "gain_1stage",
            MacroSpecificationSource::expression(
                "gm_ota__xota_1stage * ro_ota__xota_1stage",
            ),
            MacroSpecificationBounds::unbounded(),
        ))
}

fn top_path_input(spec: PdkSpec) -> MacroHierarchyPathInput {
    let mut input = MacroHierarchyPathInput::new();
    input
        .register_design_variable_override(
            "vout_1stage",
            PrimitiveBuildValue::Vector(linspace(spec.vout_1stage_start, spec.vout_1stage_stop, VOUT_1STAGE_POINTS)),
        )
        .expect("top-level VOUT_1STAGE is registered once");
    input
}

fn ota_1stage_ports() -> Vec<MacroPort> {
    vec![
        MacroPort::new("VINP", MacroPortRole::Input),
        MacroPort::new("VINN", MacroPortRole::Input),
        MacroPort::new("VOUT", MacroPortRole::Output),
        MacroPort::new("IBIAS", MacroPortRole::Bias),
        MacroPort::new("VDD", MacroPortRole::Supply),
        MacroPort::new("VSS", MacroPortRole::Ground),
    ]
}

fn ota_1stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_ota", "VOUT", "VSS", "VINP", "VSS", "gm_ota")
        .resistor("ro_ota", "VOUT", "VSS", "ro_ota")
        .capacitor("c_ota", "VOUT", "VSS", "c_ota")
        .build()
}

fn ota_2stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_ota", "VOUT", "VSS", "VINP", "VSS", 1.0)
        .resistor("ro_ota", "VOUT", "VSS", 1.0)
        .capacitor("c_ota", "VOUT", "VSS", 1.0)
        .build()
}

fn ota_1stage_seed(spec: PdkSpec) -> MacroCompactSeedSet {
    MacroCompactSeedSet::aligned([
        ("gm_ota", vec![1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]),
        ("ro_ota", vec![1e3, 1e3, 1.0e3, 1e3, 1e4, 1e4, 1e4, 1e4, 1e5, 1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1e6, 1e7, 1e7, 1e7, 1e7]),
        ("c_ota", vec![1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13]),
    ])
}

fn ac_analysis() -> AcAnalysis {
    AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT").with_polarity(TransferPolarity::Positive),
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
                min_bandwidth_3db_hz: None,
                min_unity_gain_hz: None,
                min_phase_margin_deg: None,
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

fn common_source_input(spec: PdkSpec) -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(spec.tail_current),
        ),
        (
            "VIN".to_owned(), 
            PrimitiveBuildValue::Vector(linspace(spec.vout_1stage_start, spec.vout_1stage_stop, VOUT_1STAGE_POINTS)),
        ),
        (
            "VOUT".to_owned(),
            PrimitiveBuildValue::Vector(linspace(spec.vout_start, spec.vout_stop, VOUT_POINTS)),
        ),
        (
            "VDD".to_owned(), 
            PrimitiveBuildValue::Scalar(spec.vdd)),
    ]))
}

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
    let candidates = [IHP_SPEC];
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

fn cli_options() -> Result<CliOptions, io::Error> {
let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_2stage"));
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

fn print_results(result: &MacroHierarchyExplorationResult) -> Result<(), io::Error> {
    println!("\nExploration summaries");
    for (path, node) in result.nodes() {
        match node.status() {
            MacroHierarchyNodeStatus::PreviewFinalized => {
                if let Some(final_result) = node.result() {
                    print_exploration_summary(
                        &format!("{path} [preview/final]"),
                        final_result.statistics(),
                        final_result.execution_report(),
                    );
                }
            }
            MacroHierarchyNodeStatus::PreviewRejected => {
                if let Some(final_result) = node.result() {
                    print_exploration_summary(
                        &format!("{path} [preview/rejected]"),
                        final_result.statistics(),
                        final_result.execution_report(),
                    );
                }
            }
            _ => {
                if let Some(preview) = node.preview() {
                    print_exploration_summary(
                        &format!("{path} [preview]"),
                        preview.statistics(),
                        preview.execution_report(),
                    );
                }
                if let Some(final_result) = node.result() {
                    print_exploration_summary(
                        &format!("{path} [final]"),
                        final_result.statistics(),
                        final_result.execution_report(),
                    );
                }
            }
        }
    }

    println!("\nAccepted two-stage OTA solutions");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>9} {:>11} {:>11} {:>11} {:>12} {:>9} {:>9} {:>8} {:>5} {:>8} {:>8} {:>8} {:>11} {:>12} {:>12} {:>9}",
        "root", "ota_idx", "cs_idx", "v1_v", "vout_v", "gm_ota_s", "ro_ota_ohm",
        "c_ota_f", "gain_1stage", "w_um", "wf_um", "l_um", "nf", "vbs", "vgs", "vds",
        "gain_db", "f3db_hz", "ugf_hz", "pm_deg"
    );

    for accepted_index in 0..result.root_result().accepted().len() {
        let selection = result.selection(accepted_index).map_err(io::Error::other)?;
        let root = selection
            .node(result.root_path())
            .ok_or_else(|| io::Error::other("selection has no root node"))?;
        let blackbox = selection
            .instances()
            .find(|instance| instance.instance() == OTA_1STAGE_INSTANCE)
            .ok_or_else(|| io::Error::other("selection has no ota_1stage compact candidate"))?;
        let common_source = selection
            .instances()
            .find(|instance| instance.instance() == COMMON_SOURCE_INSTANCE)
            .ok_or_else(|| io::Error::other("selection has no common-source candidate"))?;
        let metrics = &root
            .ac_outcome(OTA_2STAGE_TB)
            .ok_or_else(|| io::Error::other("selected top result has no AC outcome"))?
            .metrics;

        println!(
            "{:<6} {:>8} {:>8} {:>10.4} {:>9.4} {:>11.4e} {:>11.4e} {:>11.4e} {:>12.4} {:>9.4} {:>9.4} {:>8.4} {:>5.0} {:>8.4} {:>8.4} {:>8.4} {:>11.4} {:>12.4e} {:>12.4e} {:>9.4}",
            accepted_index,
            blackbox.candidate_index(),
            common_source.candidate_index(),
            required_value(&common_source, "xcs.vin")?,
            required_value(&common_source, "xcs.vout")?,
            required_value(&blackbox, "gm_ota__xota_1stage")?,
            required_value(&blackbox, "ro_ota__xota_1stage")?,
            required_value(&blackbox, "c_ota__xota_1stage")?,
            root.specification_value("gain_1stage")
                .ok_or_else(|| io::Error::other("selected top result has no gain_1stage value"))?,
            required_value(&common_source, "width__xcs__m1")? * 1.0e6,
            required_value(&common_source, "finger_width__xcs__m1")? * 1.0e6,
            required_value(&common_source, "length__xcs__m1")? * 1.0e6,
            required_value(&common_source, "nf__xcs__m1")?,
            required_value(&common_source, "vbs__xcs__m1")?,
            required_value(&common_source, "vgs__xcs__m1")?,
            required_value(&common_source, "vds__xcs__m1")?,
            metric(metrics.dc_gain_db, "DC gain")?,
            metric(metrics.bandwidth_3db_hz, "bandwidth")?,
            metric(metrics.unity_gain_hz, "UGF")?,
            metric(metrics.phase_margin_deg, "phase margin")?,
        );
    }
    Ok(())
}

fn print_exploration_summary(
    label: &str,
    statistics: &MacroExplorationStatistics,
    execution: &MacroExecutionReport,
) {
    println!(
        "  {label}: compatible={}, accepted={}, rejected={}, frequency_evaluations={}",
        statistics.compatible_candidates(),
        statistics.accepted_candidates(),
        statistics.rejected_candidates(),
        statistics.frequency_evaluations(),
    );
    for testbench in statistics.testbenches() {
        let rejections = testbench.rejections();
        println!(
            "    {}: evaluated={}, physical_domain={}, dc_gain={}, f3db={}, ugf={}, phase_margin={}, frequency_evaluations={}",
            testbench.testbench(),
            testbench.evaluated_candidates(),
            testbench.physical_domain_rejections(),
            rejections.count(AcMetric::DcGainDb),
            rejections.count(AcMetric::Bandwidth3DbHz),
            rejections.count(AcMetric::UnityGainHz),
            rejections.count(AcMetric::PhaseMarginDeg),
            testbench.frequency_evaluations(),
        );
    }
    for specification in statistics.specifications() {
        println!(
            "    specification {}: evaluated={}, rejected={}",
            specification.specification(),
            specification.evaluated_candidates(),
            specification.rejected_candidates(),
        );
    }
    println!(
        "    execution: candidate_build={:?}, preparation={:?}, sequential={:?}, parallel={:?}, tail={:?}, batches={}, survivors={}, total={:?}",
        execution.candidate_build(),
        execution.testbench_preparation(),
        execution.sequential_evaluation(),
        execution.parallel_electrical_analysis(),
        execution.sequential_tail(),
        execution.electrical_batches(),
        execution.electrical_survivors(),
        execution.total(),
    );
}

fn required_value(
    instance: &shapeic_core::macro_model::MacroHierarchySelectedInstance<'_>,
    column: &str,
) -> Result<f64, io::Error> {
    instance
        .value(column)
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other(format!("{} has no finite '{column}'", instance.path())))
}

fn metric(value: Option<f64>, name: &str) -> Result<f64, io::Error> {
    value
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other(format!("AC outcome has no finite {name}")))
}
