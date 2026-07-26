use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use num_complex::Complex64;
use shapeic_core::analysis::{
    AcMetrics, AnalysisMode, AnalysisTargets, TargetAssessment, TargetFailure, TargetMetric,
};
use shapeic_layout::{LayoutError, PhysicalLookupTable, PhysicalPoint, PortAdmittance};
use shapeic_lut::{
    Axis, DeviceLut, Expr, LookupTable, LutError, MosCapacitanceMatrix, MosExtrinsicCapacitances,
    OperatingPoint,
};
use shapeic_mna::evaluation::PreparedRealEvaluator;
use shapeic_mna::mna::{MnaResult, mna, mna_solve, stamp_port_admittance};
use symbolica::domains::atom::AtomField;
use symbolica::domains::float::Complex;
use symbolica::prelude::{Atom, AtomCore, Matrix};

const TAIL_CURRENT: f64 = 20.0e-6;
const LENGTHS: [f64; 5] = [0.4e-6, 0.8e-6, 1.6e-6, 3.2e-6, 6.4e-6];
const VOUT_DC: f64 = 1.0;
const VDD_DC: f64 = 1.5;
const VG_DC: f64 = 0.9;
const SOURCE_VOLTAGE_START: f64 = 0.65;
const SOURCE_VOLTAGE_STOP: f64 = 0.8;
const SOURCE_VOLTAGE_POINTS: usize = 1000;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const PHYSICAL_LAYOUT_POLICY: &str = "symmetric-adjacent-with-edge-dummies-v3";
const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
const AC_POINTS_PER_DECADE: usize = 20;
// Keep symbolic capacitance coefficients near unity without changing sC.
const AC_S_NORMALIZATION: f64 = 1.0e12;
const DC_GAIN_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];
const ANALYSIS_MODE: AnalysisMode = AnalysisMode::Prune;
const ANALYSIS_TARGETS: AnalysisTargets = AnalysisTargets {
    min_dc_gain_db: Some(27.0),
    min_bandwidth_3db_hz: Some(100.0e4),
    min_unity_gain_hz: Some(1.0e7),
    min_phase_margin_deg: Some(45.0),
};

type SmallSignalParameters = Vec<(String, f64)>;

struct SizedBlock {
    parameters: SmallSignalParameters,
    sizing: SizingSummary,
    intrinsic_capacitance: MosCapacitanceMatrix,
    extrinsic_capacitance: MosExtrinsicCapacitances,
}

#[derive(Clone, Copy)]
struct SizingSummary {
    length: f64,
    finger_width: f64,
    nf: u32,
    requested_current: f64,
    predicted_current: f64,
    current_error: f64,
}

impl SizingSummary {
    fn relative_error_percent(self) -> f64 {
        let error = 100.0 * self.current_error / self.requested_current;
        if error.abs() < 0.5e-6 { 0.0 } else { error }
    }
}

struct SweepResult {
    diff_length: f64,
    mirror_length: f64,
    source_voltage: f64,
    diff_vgs: f64,
    diff_vds: f64,
    diff_pair: SizingSummary,
    current_mirror: SizingSummary,
    gain: f64,
    dc_targets: TargetAssessment,
    electrical_ac: StageEvaluation<AcMetrics>,
    physical: Option<StageEvaluation<AcMetrics>>,
}

enum StageEvaluation<T> {
    Evaluated { value: T, targets: TargetAssessment },
    Skipped { reason: SkipReason },
    Excluded { reason: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkipReason {
    DcTargets,
    ElectricalTargets,
}

impl SkipReason {
    const fn label(self) -> &'static str {
        match self {
            Self::DcTargets => "DC target",
            Self::ElectricalTargets => "electrical targets",
        }
    }
}

struct DiffSweepPoint {
    length: f64,
    source_voltage: f64,
    vgs: f64,
    vds: f64,
    block: SizedBlock,
}

struct TimingSummary {
    nmos_lut_load: Duration,
    pmos_lut_load: Duration,
    symbolic_mna: Duration,
    dc_evaluator_preparation: Duration,
    lut_sizing: Duration,
    numeric_mna: Duration,
    electrical_ac_evaluation: Duration,
    physical_lut_load: Option<Duration>,
    physical_evaluation: Option<Duration>,
    total: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StageCounts {
    evaluated: usize,
    passed: usize,
    target_failed: usize,
    skipped: usize,
    excluded: usize,
}

struct EvaluationSummary {
    total: usize,
    dc_passed: usize,
    dc_failed: usize,
    electrical: StageCounts,
    physical: Option<StageCounts>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    ANALYSIS_TARGETS.validate()?;
    let (nmos_path, pmos_path, physical_path) = lut_paths()?;

    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let nmos_lut_load = stage_start.elapsed();
    let stage_start = Instant::now();
    let pmos_table = LookupTable::open(pmos_path)?;
    let pmos_lut_load = stage_start.elapsed();
    let (physical_table, physical_lut_load) = if let Some(path) = physical_path {
        let stage_start = Instant::now();
        let table = PhysicalLookupTable::open(path)?;
        validate_physical_layout_policy(&table)?;
        (Some(table), Some(stage_start.elapsed()))
    } else {
        (None, None)
    };
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;
    validate_intrinsic_capacitance_parameters(nmos)?;
    validate_intrinsic_capacitance_parameters(pmos)?;
    let lengths = electrical_lengths(nmos, pmos)?;
    let branch_current = TAIL_CURRENT / 2.0;

    let stage_start = Instant::now();
    let base_system = ota_mna()?;
    let gain_expression = ota_transfer_expression(&base_system, &base_system.a)?;
    let symbolic_mna = stage_start.elapsed();
    let stage_start = Instant::now();
    let mut dc_gain_evaluator =
        PreparedRealEvaluator::new(&gain_expression, &DC_GAIN_PARAMETER_ORDER).map_err(
            |error| io::Error::other(format!("could not prepare OTA DC gain evaluator: {error}")),
        )?;
    let dc_evaluator_preparation = stage_start.elapsed();
    let source_voltages = linspace(
        SOURCE_VOLTAGE_START,
        SOURCE_VOLTAGE_STOP,
        SOURCE_VOLTAGE_POINTS,
    );
    let gate_source_voltages = source_voltages
        .iter()
        .map(|source_voltage| normalize_voltage(VG_DC - source_voltage))
        .collect::<Vec<_>>();
    let drain_source_voltages = source_voltages
        .iter()
        .map(|source_voltage| normalize_voltage(VOUT_DC - source_voltage))
        .collect::<Vec<_>>();

    let stage_start = Instant::now();
    let mut diff_pairs = Vec::with_capacity(lengths.len() * source_voltages.len());
    for &length in &lengths {
        for (index, &source_voltage) in source_voltages.iter().enumerate() {
            let vgs = gate_source_voltages[index];
            let vds = drain_source_voltages[index];
            let block = simple_diff_pair(
                nmos,
                &OperatingPoint::new(length, -source_voltage, vgs, vds),
                branch_current,
            )?;
            diff_pairs.push(DiffSweepPoint {
                length,
                source_voltage,
                vgs,
                vds,
                block,
            });
        }
    }
    let current_mirrors = lengths
        .iter()
        .copied()
        .map(|length| {
            current_mirror(
                pmos,
                &OperatingPoint::new(length, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC),
                branch_current,
            )
            .map(|parameters| (length, parameters))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lut_sizing = stage_start.elapsed();
    let mut results = Vec::with_capacity(diff_pairs.len() * current_mirrors.len());

    let mut numeric_mna = Duration::ZERO;
    let mut electrical_ac_evaluation = Duration::ZERO;
    let mut physical_evaluation = physical_table.as_ref().map(|_| Duration::ZERO);
    for diff_point in &diff_pairs {
        for (mirror_length, current_mirror) in &current_mirrors {
            let stage_start = Instant::now();
            let gain =
                evaluate_ota_gain(&mut dc_gain_evaluator, &diff_point.block, current_mirror)?;
            numeric_mna += stage_start.elapsed();
            let dc_targets = ANALYSIS_TARGETS.assess_dc_gain(gain_db(gain));

            let electrical_ac = if ANALYSIS_MODE.should_continue(&dc_targets) {
                let stage_start = Instant::now();
                let metrics =
                    evaluate_electrical_ac(&base_system, &diff_point.block, current_mirror)
                        .map_err(|error| {
                            io::Error::other(format!(
                                "electrical AC evaluation failed for Ldp={:.6e}, Lcm={:.6e}, \
                                 VS={:.6e}, Wf_dp={:.6e}, nf_dp={}, Wf_cm={:.6e}, nf_cm={}: {error}",
                                diff_point.length,
                                mirror_length,
                                diff_point.source_voltage,
                                diff_point.block.sizing.finger_width,
                                diff_point.block.sizing.nf,
                                current_mirror.sizing.finger_width,
                                current_mirror.sizing.nf,
                            ))
                        })?;
                electrical_ac_evaluation += stage_start.elapsed();
                StageEvaluation::Evaluated {
                    targets: ANALYSIS_TARGETS.assess_frequency_metrics(&metrics),
                    value: metrics,
                }
            } else {
                StageEvaluation::Skipped {
                    reason: SkipReason::DcTargets,
                }
            };

            let physical = if let Some(table) = &physical_table {
                let skip_reason = match &electrical_ac {
                    StageEvaluation::Evaluated { targets, .. }
                        if !ANALYSIS_MODE.should_continue(targets) =>
                    {
                        Some(SkipReason::ElectricalTargets)
                    }
                    StageEvaluation::Skipped { reason } => Some(*reason),
                    StageEvaluation::Excluded { .. } => Some(SkipReason::ElectricalTargets),
                    StageEvaluation::Evaluated { .. } => None,
                };
                if let Some(reason) = skip_reason {
                    Some(StageEvaluation::Skipped { reason })
                } else {
                    let stage_start = Instant::now();
                    let evaluation = evaluate_layout_aware(
                        &base_system,
                        table,
                        &diff_point.block,
                        current_mirror,
                    );
                    if let Some(duration) = physical_evaluation.as_mut() {
                        *duration += stage_start.elapsed();
                    }
                    Some(match evaluation {
                        Ok(metrics) => StageEvaluation::Evaluated {
                            targets: ANALYSIS_TARGETS.assess_all(&metrics),
                            value: metrics,
                        },
                        Err(LayoutAwareError::Excluded(reason)) => {
                            StageEvaluation::Excluded { reason }
                        }
                        Err(LayoutAwareError::Fatal(error)) => {
                            return Err(io::Error::other(format!(
                                "layout-aware evaluation failed for Ldp={:.6e}, Lcm={:.6e}, \
                                 VS={:.6e}, Wf_dp={:.6e}, nf_dp={}, Wf_cm={:.6e}, nf_cm={}: {error}",
                                diff_point.length,
                                mirror_length,
                                diff_point.source_voltage,
                                diff_point.block.sizing.finger_width,
                                diff_point.block.sizing.nf,
                                current_mirror.sizing.finger_width,
                                current_mirror.sizing.nf,
                            ))
                            .into());
                        }
                    })
                }
            } else {
                None
            };

            results.push(SweepResult {
                diff_length: diff_point.length,
                mirror_length: *mirror_length,
                source_voltage: diff_point.source_voltage,
                diff_vgs: diff_point.vgs,
                diff_vds: diff_point.vds,
                diff_pair: diff_point.block.sizing,
                current_mirror: current_mirror.sizing,
                gain,
                dc_targets,
                electrical_ac,
                physical,
            });
        }
    }
    let timings = TimingSummary {
        nmos_lut_load,
        pmos_lut_load,
        symbolic_mna,
        dc_evaluator_preparation,
        lut_sizing,
        numeric_mna,
        electrical_ac_evaluation,
        physical_lut_load,
        physical_evaluation,
        total: total_start.elapsed(),
    };
    let summary = evaluation_summary(&results, physical_table.is_some());

    print_target_configuration();
    print_evaluation_summary(&summary);
    print_gain_table(&results);
    print_electrical_ac_table(&results);
    if physical_table.is_some() {
        print_physical_table(&results);
        print_ac_delta_table(&results);
    }
    print_timing_table(&timings);
    Ok(())
}

fn simple_diff_pair(
    model: &DeviceLut,
    point: &OperatingPoint,
    branch_current: f64,
) -> Result<SizedBlock, LutError> {
    let expressions = sizing_expressions();
    let sizing = model.size_for_current(point, branch_current, &expressions)?;
    let nf = f64::from(sizing.nf);
    let gm = sizing.values[0] * nf;
    let gds = sizing.values[1] * nf;
    let intrinsic_capacitance = capacitance_from_sizing(&sizing)?;
    let extrinsic_capacitance = extrinsic_capacitance_from_sizing(model, &sizing)?;

    Ok(SizedBlock {
        parameters: vec![
            ("g_gm_xdp".to_owned(), gm),
            ("r_gds_xdp".to_owned(), 1.0 / gds),
        ],
        sizing: sizing_summary(&sizing),
        intrinsic_capacitance,
        extrinsic_capacitance,
    })
}

fn current_mirror(
    model: &DeviceLut,
    point: &OperatingPoint,
    branch_current: f64,
) -> Result<SizedBlock, LutError> {
    let expressions = sizing_expressions();
    let sizing = model.size_for_current(point, branch_current, &expressions)?;
    let nf = f64::from(sizing.nf);
    let gm = sizing.values[0] * nf;
    let gds = sizing.values[1] * nf;
    let intrinsic_capacitance = capacitance_from_sizing(&sizing)?;
    let extrinsic_capacitance = extrinsic_capacitance_from_sizing(model, &sizing)?;

    Ok(SizedBlock {
        parameters: vec![
            ("g_gm_xcm".to_owned(), gm),
            ("r_gds_xcm".to_owned(), 1.0 / gds),
        ],
        sizing: sizing_summary(&sizing),
        intrinsic_capacitance,
        extrinsic_capacitance,
    })
}

fn sizing_expressions() -> Vec<Expr> {
    let mut expressions = vec![Expr::parameter("gm"), Expr::parameter("gds")];
    expressions.extend(
        MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
            .into_iter()
            .map(Expr::parameter),
    );
    expressions
}

fn capacitance_from_sizing(
    sizing: &shapeic_lut::CurrentSizingResult,
) -> Result<MosCapacitanceMatrix, LutError> {
    MosCapacitanceMatrix::from_flat(&sizing.values[2..])
        .map(|matrix| matrix.scaled(f64::from(sizing.nf)))
}

fn extrinsic_capacitance_from_sizing(
    model: &DeviceLut,
    sizing: &shapeic_lut::CurrentSizingResult,
) -> Result<MosExtrinsicCapacitances, LutError> {
    sizing
        .extrinsic_capacitances
        .ok_or_else(|| LutError::ExtrinsicCapacitanceSamplesUnavailable {
            model: model.name().to_owned(),
        })
}

fn validate_intrinsic_capacitance_parameters(model: &DeviceLut) -> Result<(), io::Error> {
    let missing = missing_intrinsic_capacitance_parameters(model.parameter_names());
    if missing.is_empty() {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "OTA AC analysis requires the nine independent intrinsic capacitance coefficients for \
         model '{}'; missing parameters: {}. Regenerate the electrical LUT with the updated \
         configuration",
        model.name(),
        missing.join(", "),
    )))
}

fn validate_physical_layout_policy(table: &PhysicalLookupTable) -> Result<(), io::Error> {
    let found = &table.metadata().layout_policy;
    if found == PHYSICAL_LAYOUT_POLICY {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "OTA physical LUT uses layout policy '{found}', expected \
         '{PHYSICAL_LAYOUT_POLICY}'. Regenerate it after the multifinger terminal-bus correction"
    )))
}

fn missing_intrinsic_capacitance_parameters(parameter_names: &[String]) -> Vec<&'static str> {
    MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
        .into_iter()
        .filter(|required| !parameter_names.iter().any(|name| name == required))
        .collect()
}

fn sizing_summary(sizing: &shapeic_lut::CurrentSizingResult) -> SizingSummary {
    SizingSummary {
        length: sizing.point.operating_point.length,
        finger_width: sizing.point.finger_width,
        nf: sizing.nf,
        requested_current: sizing.requested_current,
        predicted_current: sizing.predicted_current,
        current_error: sizing.current_error,
    }
}

fn electrical_lengths(nmos: &DeviceLut, pmos: &DeviceLut) -> Result<Vec<f64>, io::Error> {
    let contains = |model: &DeviceLut, value: f64| {
        let axis = model.axis(Axis::Length);
        let minimum = axis[0].min(axis[axis.len() - 1]);
        let maximum = axis[0].max(axis[axis.len() - 1]);
        value >= minimum && value <= maximum
    };
    let values = LENGTHS
        .into_iter()
        .filter(|value| contains(nmos, *value) && contains(pmos, *value))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(io::Error::other(
            "NMOS and PMOS LUTs do not cover any configured OTA length",
        ));
    }
    Ok(values)
}

fn ota_mna() -> Result<MnaResult, io::Error> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spice_dir = manifest.join("examples/ota_4t");
    let output_dir = manifest.join("../target/shapeic-ota-4t");
    mna(&spice_dir, &output_dir, "ota_4t")
        .map_err(|error| io::Error::other(format!("could not build OTA MNA: {error:?}")))
}

fn ota_transfer_expression(
    system: &MnaResult,
    matrix: &Matrix<AtomField>,
) -> Result<Atom, io::Error> {
    let output_row = ota_output_row(system)?;
    let output_row = u32::try_from(output_row)
        .map_err(|_| io::Error::other("VOUT row does not fit the MNA matrix index"))?;
    let solution = mna_solve(matrix, &system.x, &system.z)
        .map_err(|error| io::Error::other(format!("could not solve OTA MNA: {error:?}")))?
        .solution;
    Ok(solution[(output_row, 0)].clone())
}

fn ota_output_row(system: &MnaResult) -> Result<usize, io::Error> {
    let output_node = system
        .nodes
        .nodes
        .get("VOUT")
        .copied()
        .ok_or_else(|| io::Error::other("OTA netlist has no VOUT node"))?;
    output_node
        .checked_sub(1)
        .ok_or_else(|| io::Error::other("VOUT cannot be the ground node"))
}

enum LayoutAwareError {
    Excluded(String),
    Fatal(Box<dyn Error>),
}

fn evaluate_electrical_ac(
    base_system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
) -> Result<AcMetrics, io::Error> {
    let mut matrix = base_system.a.clone();
    stamp_compact_model_devices(
        &mut matrix,
        base_system,
        diff_pair,
        current_mirror,
        AC_S_NORMALIZATION,
    )?;
    let parameters = diff_pair
        .parameters
        .iter()
        .chain(&current_mirror.parameters)
        .cloned()
        .collect::<Vec<_>>();
    evaluate_numeric_ac_metrics(&matrix, base_system, &parameters)
}

fn evaluate_layout_aware(
    base_system: &MnaResult,
    table: &PhysicalLookupTable,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
) -> Result<AcMetrics, LayoutAwareError> {
    let diff_physical = query_physical(table, "simplediffpair", diff_pair.sizing)?;
    let mirror_physical = query_physical(table, "currentmirror", current_mirror.sizing)?;

    let mut matrix = base_system.a.clone();
    stamp_physical(
        &mut matrix,
        base_system,
        &diff_physical,
        diff_pair_physical_connections(),
        1.0,
    )
    .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))?;
    stamp_physical(
        &mut matrix,
        base_system,
        &mirror_physical,
        [("DOUT", "VOUT"), ("DREF", "N1"), ("S", "VDD"), ("B", "VDD")],
        1.0,
    )
    .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))?;
    stamp_compact_model_devices(&mut matrix, base_system, diff_pair, current_mirror, 1.0)
        .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))?;
    let expression = ota_transfer_expression(base_system, &matrix)
        .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))?;
    let parameters = diff_pair
        .parameters
        .iter()
        .chain(&current_mirror.parameters)
        .cloned()
        .collect::<Vec<_>>();
    evaluate_ac_metrics(&expression, &parameters, 1.0)
        .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))
}

fn diff_pair_mos_connections<'a>(gate: &'a str, drain: &'a str) -> [(&'static str, &'a str); 4] {
    [("G", gate), ("D", drain), ("S", "IBIAS"), ("B", "VSS")]
}

fn diff_pair_physical_connections() -> [(&'static str, &'static str); 6] {
    [
        ("DP", "VOUT"),
        ("DN", "N1"),
        ("GP", "VINP"),
        ("GN", "VINN"),
        ("S", "IBIAS"),
        ("B", "VSS"),
    ]
}

fn stamp_compact_model_devices(
    matrix: &mut Matrix<AtomField>,
    base_system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    capacitance_scale: f64,
) -> Result<(), io::Error> {
    stamp_complete_mos_capacitance(
        matrix,
        base_system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_mos_connections("VINP", "VOUT"),
        capacitance_scale,
    )?;
    stamp_complete_mos_capacitance(
        matrix,
        base_system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_mos_connections("VINN", "N1"),
        capacitance_scale,
    )?;
    stamp_complete_mos_capacitance(
        matrix,
        base_system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD")],
        capacitance_scale,
    )?;
    stamp_complete_mos_capacitance(
        matrix,
        base_system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD")],
        capacitance_scale,
    )
}

fn query_physical(
    table: &PhysicalLookupTable,
    primitive: &str,
    sizing: SizingSummary,
) -> Result<PortAdmittance, LayoutAwareError> {
    let model = table
        .primitive(primitive)
        .map_err(|error| LayoutAwareError::Fatal(Box::new(error)))?;
    match model.query(PhysicalPoint::new(
        sizing.length,
        sizing.finger_width,
        sizing.nf,
    )) {
        Ok(value) => Ok(value),
        Err(LayoutError::OutOfPhysicalRange { axis, value, .. }) => Err(
            LayoutAwareError::Excluded(format!("{primitive}: {axis}={value:.4e} outside LUT")),
        ),
        Err(LayoutError::InvalidFingerCount) => Err(LayoutAwareError::Excluded(format!(
            "{primitive}: invalid nf={}",
            sizing.nf
        ))),
        Err(error) => Err(LayoutAwareError::Fatal(Box::new(error))),
    }
}

fn stamp_physical<const N: usize>(
    matrix: &mut Matrix<AtomField>,
    system: &MnaResult,
    admittance: &PortAdmittance,
    connections: [(&str, &str); N],
    capacitance_scale: f64,
) -> Result<(), io::Error> {
    let connections = connections
        .into_iter()
        .map(|(port, node)| (port.to_owned(), node.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let normalized_capacitance = admittance
        .capacitance
        .mapv(|value| value * capacitance_scale);
    stamp_port_admittance(
        matrix,
        &system.nodes,
        &admittance.ports,
        &connections,
        admittance.conductance.view(),
        normalized_capacitance.view(),
    )
    .map_err(|error| io::Error::other(format!("could not stamp physical primitive: {error:?}")))
}

fn stamp_complete_mos_capacitance<const N: usize>(
    matrix: &mut Matrix<AtomField>,
    system: &MnaResult,
    intrinsic: MosCapacitanceMatrix,
    extrinsic: MosExtrinsicCapacitances,
    connections: [(&str, &str); N],
    capacitance_scale: f64,
) -> Result<(), io::Error> {
    stamp_intrinsic(
        matrix,
        system,
        MosCapacitanceMatrix {
            values: combined_capacitance_matrix(intrinsic, extrinsic),
        },
        connections,
        capacitance_scale,
    )
}

fn combined_capacitance_matrix(
    intrinsic: MosCapacitanceMatrix,
    extrinsic: MosExtrinsicCapacitances,
) -> [[f64; 4]; 4] {
    let mut matrix = intrinsic.values;
    add_passive_capacitance(&mut matrix, 0, 2, extrinsic.cgsol);
    add_passive_capacitance(&mut matrix, 0, 1, extrinsic.cgdol);
    add_passive_capacitance(&mut matrix, 2, 3, extrinsic.cjs);
    add_passive_capacitance(&mut matrix, 1, 3, extrinsic.cjd);
    matrix
}

fn add_passive_capacitance(
    matrix: &mut [[f64; 4]; 4],
    positive: usize,
    negative: usize,
    capacitance: f64,
) {
    matrix[positive][positive] += capacitance;
    matrix[negative][negative] += capacitance;
    matrix[positive][negative] -= capacitance;
    matrix[negative][positive] -= capacitance;
}

fn stamp_intrinsic<const N: usize>(
    matrix: &mut Matrix<AtomField>,
    system: &MnaResult,
    capacitance: MosCapacitanceMatrix,
    connections: [(&str, &str); N],
    capacitance_scale: f64,
) -> Result<(), io::Error> {
    let flat = capacitance.values.into_iter().flatten().collect::<Vec<_>>();
    let capacitance = Array2::from_shape_vec((4, 4), flat)
        .expect("fixed 4x4 matrix shape")
        .mapv(|value| value * capacitance_scale);
    let conductance = Array2::<f64>::zeros((4, 4));
    let ports = MosCapacitanceMatrix::TERMINALS.map(str::to_owned).to_vec();
    let connections = connections
        .into_iter()
        .map(|(port, node)| (port.to_owned(), node.to_owned()))
        .collect::<BTreeMap<_, _>>();
    stamp_port_admittance(
        matrix,
        &system.nodes,
        &ports,
        &connections,
        conductance.view(),
        capacitance.view(),
    )
    .map_err(|error| {
        io::Error::other(format!(
            "could not stamp intrinsic MOS capacitance: {error:?}"
        ))
    })
}

fn evaluate_numeric_ac_metrics(
    matrix: &Matrix<AtomField>,
    system: &MnaResult,
    parameters: &[(String, f64)],
) -> Result<AcMetrics, io::Error> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows != columns || system.z.nrows() != matrix.nrows() || system.z.ncols() != 1 {
        return Err(io::Error::other(
            "numeric AC analysis requires a square MNA and one matching RHS column",
        ));
    }

    let matrix_len = rows * columns;
    let mut expressions = Vec::with_capacity(matrix_len + rows);
    for row in 0..matrix.nrows() {
        let row_index =
            u32::try_from(row).map_err(|_| io::Error::other("OTA MNA row index exceeds u32"))?;
        for column in 0..matrix.ncols() {
            let column_index = u32::try_from(column)
                .map_err(|_| io::Error::other("OTA MNA column index exceeds u32"))?;
            expressions.push(matrix[(row_index, column_index)].clone());
        }
    }
    for row in 0..system.z.nrows() {
        let row_index =
            u32::try_from(row).map_err(|_| io::Error::other("OTA MNA row index exceeds u32"))?;
        expressions.push(system.z[(row_index, 0)].clone());
    }
    let mut symbol_map = BTreeMap::new();
    for expression in &expressions {
        for symbol in expression.get_all_symbols(false) {
            let symbol = Atom::from(symbol);
            symbol_map.entry(symbol.to_string()).or_insert(symbol);
        }
    }
    let symbols = symbol_map.into_values().collect::<Vec<_>>();
    let evaluator = Atom::evaluator_multiple(&expressions, &symbols)
        .build()
        .map_err(|error| {
            io::Error::other(format!("could not build numeric MNA evaluator: {error}"))
        })?;
    let mut evaluator = evaluator
        .map_coeff(&|coefficient| Complex::new(coefficient.re.to_f64(), coefficient.im.to_f64()));
    let parameter_map = parameters.iter().cloned().collect::<BTreeMap<_, _>>();
    let output_row = ota_output_row(system)?;
    let mut outputs = vec![Complex::new(0.0, 0.0); expressions.len()];
    let mut evaluate = |frequency_hz: f64| -> Result<Option<Complex<f64>>, io::Error> {
        let values =
            ac_evaluator_inputs(&symbols, &parameter_map, frequency_hz, AC_S_NORMALIZATION)?;
        evaluator
            .try_evaluate(&values, &mut outputs)
            .map_err(|error| {
                io::Error::other(format!(
                    "could not evaluate numeric MNA at {frequency_hz:.6e} Hz: {error}"
                ))
            })?;
        solve_numeric_mna(&outputs, matrix_len, rows, output_row, frequency_hz)
    };

    let frequencies = ac_frequencies();
    let responses = frequencies
        .iter()
        .map(|frequency| {
            evaluate(*frequency)?.ok_or_else(|| {
                io::Error::other(format!("numeric OTA MNA is singular at {frequency:.6e} Hz"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dc_response = evaluate(0.0)?.unwrap_or(responses[0]);
    Ok(ac_metrics_from_responses(
        &frequencies,
        &responses,
        dc_response,
    ))
}

fn solve_numeric_mna(
    outputs: &[Complex<f64>],
    matrix_len: usize,
    rows: usize,
    output_row: usize,
    frequency_hz: f64,
) -> Result<Option<Complex<f64>>, io::Error> {
    let expected_matrix_len = rows
        .checked_mul(rows)
        .ok_or_else(|| io::Error::other("numeric OTA MNA dimensions overflow usize"))?;
    let expected_outputs = expected_matrix_len
        .checked_add(rows)
        .ok_or_else(|| io::Error::other("numeric OTA MNA output count overflows usize"))?;
    if rows == 0
        || matrix_len != expected_matrix_len
        || outputs.len() != expected_outputs
        || output_row >= rows
    {
        return Err(io::Error::other(format!(
            "invalid numeric OTA MNA dimensions: rows={rows}, matrix_len={matrix_len}, \
             outputs={}, output_row={output_row}",
            outputs.len()
        )));
    }
    if outputs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(io::Error::other(format!(
            "numeric OTA MNA contains a non-finite coefficient at {frequency_hz:.6e} Hz"
        )));
    }
    let coefficients = outputs[..matrix_len]
        .iter()
        .map(|value| Complex64::new(value.re, value.im))
        .collect::<Vec<_>>();
    let rhs = outputs[matrix_len..]
        .iter()
        .map(|value| Complex64::new(value.re, value.im))
        .collect::<Vec<_>>();
    let solution = DMatrix::from_row_slice(rows, rows, &coefficients)
        .lu()
        .solve(&DVector::from_vec(rhs));
    let Some(solution) = solution else {
        return Ok(None);
    };
    let output = solution[output_row];
    if !output.re.is_finite() || !output.im.is_finite() {
        return Err(io::Error::other(format!(
            "numeric OTA response is not finite at {frequency_hz:.6e} Hz"
        )));
    }
    Ok(Some(Complex::new(output.re, output.im)))
}

fn evaluate_ac_metrics(
    expression: &Atom,
    parameters: &[(String, f64)],
    s_normalization: f64,
) -> Result<AcMetrics, io::Error> {
    let symbols = expression
        .get_all_symbols(false)
        .into_iter()
        .map(Atom::from)
        .collect::<Vec<_>>();
    let evaluator = expression
        .evaluator(&symbols)
        .build()
        .map_err(|error| io::Error::other(format!("could not build AC evaluator: {error}")))?;
    let mut evaluator = evaluator
        .map_coeff(&|coefficient| Complex::new(coefficient.re.to_f64(), coefficient.im.to_f64()));
    let parameter_map = parameters.iter().cloned().collect::<BTreeMap<_, _>>();
    let mut evaluate = |frequency_hz: f64| -> Result<Complex<f64>, io::Error> {
        let values = ac_evaluator_inputs(&symbols, &parameter_map, frequency_hz, s_normalization)?;
        Ok(evaluator.evaluate_single(&values))
    };

    let frequencies = ac_frequencies();
    let responses = frequencies
        .iter()
        .map(|frequency| {
            let response = evaluate(*frequency)?;
            if !is_finite_response(response) {
                return Err(io::Error::other(format!(
                    "OTA AC response is not finite at {frequency:.6e} Hz"
                )));
            }
            Ok(response)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exact_dc = evaluate(0.0)?;
    let dc_response = select_dc_response(exact_dc, responses[0]);
    Ok(ac_metrics_from_responses(
        &frequencies,
        &responses,
        dc_response,
    ))
}

fn ac_evaluator_inputs(
    symbols: &[Atom],
    parameter_map: &BTreeMap<String, f64>,
    frequency_hz: f64,
    s_normalization: f64,
) -> Result<Vec<Complex<f64>>, io::Error> {
    symbols
        .iter()
        .map(|symbol| {
            let name = symbol.to_string();
            if name == "s" {
                Ok(laplace_frequency(frequency_hz, s_normalization))
            } else {
                parameter_map
                    .get(&name)
                    .copied()
                    .map(|value| Complex::new(value, 0.0))
                    .ok_or_else(|| io::Error::other(format!("no value for MNA parameter '{name}'")))
            }
        })
        .collect()
}

fn ac_metrics_from_responses(
    frequencies: &[f64],
    amplifier_responses: &[Complex<f64>],
    amplifier_dc_response: Complex<f64>,
) -> AcMetrics {
    let loop_responses = amplifier_responses
        .iter()
        .copied()
        .map(loop_response)
        .collect::<Vec<_>>();
    let dc_gain_db = magnitude_db(loop_response(amplifier_dc_response));
    let magnitudes = loop_responses
        .iter()
        .copied()
        .map(magnitude_db)
        .collect::<Vec<_>>();
    let phases = unwrap_phases(&loop_responses);
    let bandwidth_3db_hz = crossing_frequency(frequencies, &magnitudes, dc_gain_db - 3.0)
        .map(|(frequency, _)| frequency);
    let unity = crossing_frequency(frequencies, &magnitudes, 0.0);
    let (unity_gain_hz, phase_margin_deg) = if let Some((frequency, interval)) = unity {
        let phase = interpolate_at_log_frequency(
            frequencies[interval],
            frequencies[interval + 1],
            phases[interval],
            phases[interval + 1],
            frequency,
        );
        (Some(frequency), Some(180.0 + phase))
    } else {
        (None, None)
    };
    AcMetrics {
        dc_gain_db,
        bandwidth_3db_hz,
        unity_gain_hz,
        phase_margin_deg,
    }
}

fn loop_response(amplifier_response: Complex<f64>) -> Complex<f64> {
    -amplifier_response
}

fn laplace_frequency(frequency_hz: f64, normalization: f64) -> Complex<f64> {
    Complex::new(
        0.0,
        2.0 * std::f64::consts::PI * frequency_hz / normalization,
    )
}

fn is_finite_response(value: Complex<f64>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn select_dc_response(exact_dc: Complex<f64>, low_frequency: Complex<f64>) -> Complex<f64> {
    if is_finite_response(exact_dc) {
        exact_dc
    } else {
        low_frequency
    }
}

fn ac_frequencies() -> Vec<f64> {
    let decades = (AC_MAX_HZ / AC_MIN_HZ).log10();
    let count = (decades * AC_POINTS_PER_DECADE as f64).round() as usize + 1;
    (0..count)
        .map(|index| AC_MIN_HZ * 10.0_f64.powf(index as f64 / AC_POINTS_PER_DECADE as f64))
        .collect()
}

fn magnitude_db(value: Complex<f64>) -> f64 {
    10.0 * value.norm_squared().log10()
}

fn unwrap_phases(responses: &[Complex<f64>]) -> Vec<f64> {
    let mut phases = Vec::with_capacity(responses.len());
    for response in responses {
        let mut phase = response.arg().to_degrees();
        if phases.is_empty() && phase > 90.0 {
            phase -= 360.0;
        }
        if let Some(previous) = phases.last().copied() {
            while phase - previous > 180.0 {
                phase -= 360.0;
            }
            while phase - previous < -180.0 {
                phase += 360.0;
            }
        }
        phases.push(phase);
    }
    phases
}

fn crossing_frequency(frequencies: &[f64], values: &[f64], target: f64) -> Option<(f64, usize)> {
    values.windows(2).enumerate().find_map(|(index, pair)| {
        if pair[0] >= target && pair[1] <= target && pair[0] != pair[1] {
            let weight = (target - pair[0]) / (pair[1] - pair[0]);
            let log_frequency = frequencies[index].log10()
                + weight * (frequencies[index + 1].log10() - frequencies[index].log10());
            Some((10.0_f64.powf(log_frequency), index))
        } else {
            None
        }
    })
}

fn interpolate_at_log_frequency(
    lower_frequency: f64,
    upper_frequency: f64,
    lower_value: f64,
    upper_value: f64,
    frequency: f64,
) -> f64 {
    let weight = (frequency.log10() - lower_frequency.log10())
        / (upper_frequency.log10() - lower_frequency.log10());
    lower_value + weight * (upper_value - lower_value)
}

fn evaluate_ota_gain(
    evaluator: &mut PreparedRealEvaluator,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
) -> Result<f64, io::Error> {
    let parameter = |block: &SizedBlock, name: &str| {
        block
            .parameters
            .iter()
            .find_map(|(parameter, value)| (parameter == name).then_some(*value))
            .ok_or_else(|| io::Error::other(format!("no value for MNA parameter '{name}'")))
    };
    let values = [
        parameter(diff_pair, DC_GAIN_PARAMETER_ORDER[0])?,
        parameter(diff_pair, DC_GAIN_PARAMETER_ORDER[1])?,
        parameter(current_mirror, DC_GAIN_PARAMETER_ORDER[2])?,
        parameter(current_mirror, DC_GAIN_PARAMETER_ORDER[3])?,
    ];
    let gain = evaluator
        .evaluate(&values)
        .map_err(|error| io::Error::other(format!("could not evaluate OTA gain: {error}")))?;
    if !gain.is_finite() {
        return Err(io::Error::other("OTA gain is not finite"));
    }
    Ok(gain)
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

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (count - 1) as f64;
            (0..count)
                .map(|index| {
                    if index + 1 == count {
                        stop
                    } else {
                        start + step * index as f64
                    }
                })
                .collect()
        }
    }
}

fn normalize_voltage(voltage: f64) -> f64 {
    const DECIMAL_SCALE: f64 = 1.0e12;
    (voltage * DECIMAL_SCALE).round() / DECIMAL_SCALE
}

fn gain_db(gain: f64) -> f64 {
    20.0 * gain.abs().log10()
}

fn evaluation_summary(results: &[SweepResult], physical_enabled: bool) -> EvaluationSummary {
    let dc_passed = results
        .iter()
        .filter(|result| result.dc_targets.passed())
        .count();
    EvaluationSummary {
        total: results.len(),
        dc_passed,
        dc_failed: results.len() - dc_passed,
        electrical: count_stage(results.iter().map(|result| &result.electrical_ac)),
        physical: physical_enabled
            .then(|| count_stage(results.iter().filter_map(|result| result.physical.as_ref()))),
    }
}

fn count_stage<'a, T: 'a>(
    evaluations: impl IntoIterator<Item = &'a StageEvaluation<T>>,
) -> StageCounts {
    let mut counts = StageCounts::default();
    for evaluation in evaluations {
        match evaluation {
            StageEvaluation::Evaluated { targets, .. } => {
                counts.evaluated += 1;
                if targets.passed() {
                    counts.passed += 1;
                } else {
                    counts.target_failed += 1;
                }
            }
            StageEvaluation::Skipped { .. } => counts.skipped += 1,
            StageEvaluation::Excluded { .. } => counts.excluded += 1,
        }
    }
    counts
}

fn print_target_configuration() {
    println!("Analysis targets");
    println!("  mode       = {}", analysis_mode_label(ANALYSIS_MODE));
    for (label, target) in [
        ("GainDC", ANALYSIS_TARGETS.min_dc_gain_db),
        ("f3dB", ANALYSIS_TARGETS.min_bandwidth_3db_hz),
        ("UGF", ANALYSIS_TARGETS.min_unity_gain_hz),
        ("PM", ANALYSIS_TARGETS.min_phase_margin_deg),
    ] {
        println!(
            "  {label:<10} = {}",
            target.map_or_else(|| "disabled".to_owned(), |value| format!("{value:.6e}"))
        );
    }
}

fn analysis_mode_label(mode: AnalysisMode) -> &'static str {
    match mode {
        AnalysisMode::Prune => "prune",
        AnalysisMode::FullInsight => "full-insight",
    }
}

fn print_evaluation_summary(summary: &EvaluationSummary) {
    println!();
    println!("Candidate flow");
    println!(
        "{:<12} | {:>10} | {:>10} | {:>11} | {:>10} | {:>10}",
        "Stage", "Evaluated", "Passed", "Target fail", "Skipped", "Excluded"
    );
    println!("-------------+------------+------------+-------------+------------+-----------");
    println!(
        "{:<12} | {:>10} | {:>10} | {:>11} | {:>10} | {:>10}",
        "DC gain", summary.total, summary.dc_passed, summary.dc_failed, 0, 0
    );
    print_stage_counts("Electrical", summary.electrical);
    if let Some(physical) = summary.physical {
        print_stage_counts("Physical", physical);
    }
}

fn print_stage_counts(stage: &str, counts: StageCounts) {
    println!(
        "{stage:<12} | {:>10} | {:>10} | {:>11} | {:>10} | {:>10}",
        counts.evaluated, counts.passed, counts.target_failed, counts.skipped, counts.excluded,
    );
}

fn print_gain_table(results: &[SweepResult]) {
    println!();
    println!("Electrical sizing and DC gain");
    println!(
        "{:>9} | {:>9} | {:>7} | {:>9} | {:>9} | {:>10} | {:>10} | {:>5} | {:>10} | {:>10} | {:>10} | {:>5} | {:>10} | {:>10} | {:>16} | {:>11} | status",
        "L_dp[um]",
        "L_cm[um]",
        "VS[V]",
        "VGS_dp[V]",
        "VDS_dp[V]",
        "Ireq[uA]",
        "Wf_dp[um]",
        "nf_dp",
        "Id_dp[uA]",
        "Err_dp[%]",
        "Wf_cm[um]",
        "nf_cm",
        "Id_cm[uA]",
        "Err_cm[%]",
        "Gain[V/V]",
        "Gain[dB]",
    );
    println!(
        "----------+-----------+---------+-----------+-----------+------------+------------+-------+------------+------------+------------+-------+------------+------------+------------------+-------------+------------------------------"
    );
    for result in results {
        println!(
            "{:>9.3} | {:>9.3} | {:>7.3} | {:>9.3} | {:>9.3} | {:>10.6} | {:>10.6} | {:>5} | {:>10.6} | {:>10.6} | {:>10.6} | {:>5} | {:>10.6} | {:>10.6} | {:>16.6e} | {:>11.3} | {}",
            result.diff_length * 1.0e6,
            result.mirror_length * 1.0e6,
            result.source_voltage,
            result.diff_vgs,
            result.diff_vds,
            result.diff_pair.requested_current * 1.0e6,
            result.diff_pair.finger_width * 1.0e6,
            result.diff_pair.nf,
            result.diff_pair.predicted_current * 1.0e6,
            result.diff_pair.relative_error_percent(),
            result.current_mirror.finger_width * 1.0e6,
            result.current_mirror.nf,
            result.current_mirror.predicted_current * 1.0e6,
            result.current_mirror.relative_error_percent(),
            result.gain,
            gain_db(result.gain),
            format_target_assessment(&result.dc_targets),
        );
    }
}

fn print_electrical_ac_table(results: &[SweepResult]) {
    print_ac_table_header("Electrical AC (intrinsic + compact-model extrinsic capacitances)");
    for result in results {
        print_ac_evaluation_row(result, &result.electrical_ac);
    }
}

fn print_physical_table(results: &[SweepResult]) {
    print_ac_table_header(
        "Layout-aware AC (intrinsic + compact-model extrinsic + physical parasitics)",
    );
    for result in results {
        let Some(physical) = &result.physical else {
            continue;
        };
        print_ac_evaluation_row(result, physical);
    }
}

fn print_ac_table_header(title: &str) {
    println!();
    println!("{title}");
    println!(
        "{:>9} | {:>9} | {:>7} | {:>9} | {:>5} | {:>9} | {:>5} | {:>11} | {:>11} | {:>11} | {:>9} | status",
        "L_dp[um]",
        "L_cm[um]",
        "VS[V]",
        "Wf_dp[um]",
        "nf_dp",
        "Wf_cm[um]",
        "nf_cm",
        "GainDC[dB]",
        "f3dB[Hz]",
        "UGF[Hz]",
        "PM[deg]",
    );
    println!(
        "----------+-----------+---------+-----------+-------+-----------+-------+-------------+-------------+-------------+-----------+------------------------------"
    );
}

fn print_ac_evaluation_row(result: &SweepResult, evaluation: &StageEvaluation<AcMetrics>) {
    match evaluation {
        StageEvaluation::Evaluated { value, targets } => {
            print_ac_metrics_row(result, *value, &format_target_assessment(targets));
        }
        StageEvaluation::Skipped { reason } => {
            print_empty_ac_row(result, &format!("skipped: {}", reason.label()));
        }
        StageEvaluation::Excluded { reason } => {
            print_empty_ac_row(result, &format!("excluded: {reason}"));
        }
    }
}

fn print_ac_metrics_row(result: &SweepResult, metrics: AcMetrics, status: &str) {
    println!(
        "{:>9.3} | {:>9.3} | {:>7.3} | {:>9.3} | {:>5} | {:>9.3} | {:>5} | {:>11.3} | {:>11} | {:>11} | {:>9} | {status}",
        result.diff_length * 1.0e6,
        result.mirror_length * 1.0e6,
        result.source_voltage,
        result.diff_pair.finger_width * 1.0e6,
        result.diff_pair.nf,
        result.current_mirror.finger_width * 1.0e6,
        result.current_mirror.nf,
        metrics.dc_gain_db,
        format_frequency(metrics.bandwidth_3db_hz),
        format_frequency(metrics.unity_gain_hz),
        format_optional(metrics.phase_margin_deg),
    );
}

fn print_empty_ac_row(result: &SweepResult, status: &str) {
    println!(
        "{:>9.3} | {:>9.3} | {:>7.3} | {:>9.3} | {:>5} | {:>9.3} | {:>5} | {:>11} | {:>11} | {:>11} | {:>9} | {status}",
        result.diff_length * 1.0e6,
        result.mirror_length * 1.0e6,
        result.source_voltage,
        result.diff_pair.finger_width * 1.0e6,
        result.diff_pair.nf,
        result.current_mirror.finger_width * 1.0e6,
        result.current_mirror.nf,
        "-",
        "-",
        "-",
        "-",
    );
}

fn print_ac_delta_table(results: &[SweepResult]) {
    println!();
    println!("Physical parasitic impact (physical - electrical)");
    println!(
        "{:>9} | {:>9} | {:>7} | {:>9} | {:>5} | {:>9} | {:>5} | {:>11} | {:>11} | {:>11} | {:>10} | status",
        "L_dp[um]",
        "L_cm[um]",
        "VS[V]",
        "Wf_dp[um]",
        "nf_dp",
        "Wf_cm[um]",
        "nf_cm",
        "dGain[dB]",
        "df3dB[%]",
        "dUGF[%]",
        "dPM[deg]",
    );
    println!(
        "----------+-----------+---------+-----------+-------+-----------+-------+-------------+-------------+-------------+------------+------------------------------"
    );
    for result in results {
        let Some(physical) = &result.physical else {
            continue;
        };
        let (gain, bandwidth, unity, phase, status) = match (&result.electrical_ac, physical) {
            (
                StageEvaluation::Evaluated {
                    value: electrical, ..
                },
                StageEvaluation::Evaluated {
                    value: physical,
                    targets,
                },
            ) => (
                format!("{:.3}", physical.dc_gain_db - electrical.dc_gain_db),
                format_optional(relative_change_percent(
                    electrical.bandwidth_3db_hz,
                    physical.bandwidth_3db_hz,
                )),
                format_optional(relative_change_percent(
                    electrical.unity_gain_hz,
                    physical.unity_gain_hz,
                )),
                format_optional(option_difference(
                    electrical.phase_margin_deg,
                    physical.phase_margin_deg,
                )),
                format_target_assessment(targets),
            ),
            (_, StageEvaluation::Skipped { reason }) => (
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                format!("skipped: {}", reason.label()),
            ),
            (_, StageEvaluation::Excluded { reason }) => (
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                format!("excluded: {reason}"),
            ),
            (StageEvaluation::Skipped { reason }, _) => (
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                format!("skipped: {}", reason.label()),
            ),
            (StageEvaluation::Excluded { reason }, _) => (
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                "-".to_owned(),
                format!("excluded: {reason}"),
            ),
        };
        println!(
            "{:>9.3} | {:>9.3} | {:>7.3} | {:>9.3} | {:>5} | {:>9.3} | {:>5} | {:>11} | {:>11} | {:>11} | {:>10} | {status}",
            result.diff_length * 1.0e6,
            result.mirror_length * 1.0e6,
            result.source_voltage,
            result.diff_pair.finger_width * 1.0e6,
            result.diff_pair.nf,
            result.current_mirror.finger_width * 1.0e6,
            result.current_mirror.nf,
            gain,
            bandwidth,
            unity,
            phase,
        );
    }
}

fn format_target_assessment(assessment: &TargetAssessment) -> String {
    if assessment.passed() {
        return "pass".to_owned();
    }
    let failures = assessment
        .failures()
        .iter()
        .map(format_target_failure)
        .collect::<Vec<_>>()
        .join(", ");
    format!("fail: {failures}")
}

fn format_target_failure(failure: &TargetFailure) -> String {
    let label = failure.metric.label();
    let Some(actual) = failure.actual else {
        return format!("{label} unavailable");
    };
    format!(
        "{label}={}<{}",
        format_metric_value(failure.metric, actual),
        format_metric_value(failure.metric, failure.minimum),
    )
}

fn format_metric_value(metric: TargetMetric, value: f64) -> String {
    match metric {
        TargetMetric::Bandwidth3DbHz | TargetMetric::UnityGainHz => format!("{value:.3e}"),
        TargetMetric::DcGainDb | TargetMetric::PhaseMarginDeg => format!("{value:.3}"),
    }
}

fn relative_change_percent(reference: Option<f64>, compared: Option<f64>) -> Option<f64> {
    let (reference, compared) = (reference?, compared?);
    if !reference.is_finite() || !compared.is_finite() || reference == 0.0 {
        return None;
    }
    Some(100.0 * (compared / reference - 1.0))
}

fn option_difference(reference: Option<f64>, compared: Option<f64>) -> Option<f64> {
    Some(compared? - reference?)
}

fn format_frequency(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3e}"))
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn print_timing_table(timings: &TimingSummary) {
    println!();
    println!("{:<28} | {:>12}", "Stage", "Time [ms]");
    println!("-----------------------------+-------------");
    for (stage, duration) in [
        ("NMOS LUT load", timings.nmos_lut_load),
        ("PMOS LUT load", timings.pmos_lut_load),
        ("Symbolic MNA build + solve", timings.symbolic_mna),
        ("DC evaluator preparation", timings.dc_evaluator_preparation),
        ("LUT sizing + interpolation", timings.lut_sizing),
        ("DC MNA evaluation", timings.numeric_mna),
        (
            "Electrical numeric AC MNA",
            timings.electrical_ac_evaluation,
        ),
    ] {
        println!("{stage:<28} | {:>12.3}", duration.as_secs_f64() * 1.0e3);
    }
    if let Some(duration) = timings.physical_lut_load {
        println!(
            "{:<28} | {:>12.3}",
            "Physical LUT load",
            duration.as_secs_f64() * 1.0e3
        );
    }
    if let Some(duration) = timings.physical_evaluation {
        println!(
            "{:<28} | {:>12.3}",
            "Physical query + AC MNA",
            duration.as_secs_f64() * 1.0e3
        );
    }
    println!(
        "{:<28} | {:>12.3}",
        "Total computation",
        timings.total.as_secs_f64() * 1.0e3
    );
}

#[cfg(test)]
mod tests {
    use super::{
        AC_MAX_HZ, AC_MIN_HZ, AC_POINTS_PER_DECADE, AC_S_NORMALIZATION, ANALYSIS_TARGETS,
        AcMetrics, SOURCE_VOLTAGE_POINTS, SOURCE_VOLTAGE_START, SOURCE_VOLTAGE_STOP, SkipReason,
        StageCounts, StageEvaluation, VG_DC, VOUT_DC, ac_frequencies, ac_metrics_from_responses,
        combined_capacitance_matrix, count_stage, crossing_frequency, diff_pair_mos_connections,
        diff_pair_physical_connections, format_target_assessment, gain_db, laplace_frequency,
        linspace, missing_intrinsic_capacitance_parameters, normalize_voltage, option_difference,
        relative_change_percent, select_dc_response, solve_numeric_mna,
    };
    use shapeic_core::analysis::AnalysisTargets;
    use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};
    use symbolica::domains::float::Complex;

    #[test]
    fn linspace_handles_empty_single_and_regular_ranges() {
        assert!(linspace(0.1, 0.5, 0).is_empty());
        assert_eq!(linspace(0.1, 0.5, 1), vec![0.1]);

        let values = linspace(
            SOURCE_VOLTAGE_START,
            SOURCE_VOLTAGE_STOP,
            SOURCE_VOLTAGE_POINTS,
        );
        assert_eq!(values.len(), SOURCE_VOLTAGE_POINTS);
        assert_eq!(values[0], SOURCE_VOLTAGE_START);
        assert_eq!(values[SOURCE_VOLTAGE_POINTS - 1], SOURCE_VOLTAGE_STOP);

        let drain_source_voltages = values
            .iter()
            .map(|source_voltage| normalize_voltage(VOUT_DC - source_voltage))
            .collect::<Vec<_>>();
        let gate_source_voltages = values
            .iter()
            .map(|source_voltage| normalize_voltage(VG_DC - source_voltage))
            .collect::<Vec<_>>();
        assert_eq!(drain_source_voltages[0], 0.35);
        assert_eq!(drain_source_voltages[SOURCE_VOLTAGE_POINTS - 1], 0.2);
        assert_eq!(gate_source_voltages[0], 0.25);
        assert_eq!(gate_source_voltages[SOURCE_VOLTAGE_POINTS - 1], 0.1);
    }

    #[test]
    fn reports_missing_intrinsic_capacitance_parameters() {
        let mut parameters = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert!(missing_intrinsic_capacitance_parameters(&parameters).is_empty());

        parameters.retain(|parameter| parameter != "cgd" && parameter != "css");
        assert_eq!(
            missing_intrinsic_capacitance_parameters(&parameters),
            vec!["cgd", "css"]
        );
    }

    #[test]
    fn compact_model_diff_pair_bulk_is_connected_to_vss() {
        let connections = diff_pair_mos_connections("VINP", "VOUT");
        assert_eq!(connections[2], ("S", "IBIAS"));
        assert_eq!(connections[3], ("B", "VSS"));
    }

    #[test]
    fn physical_diff_pair_bulk_is_connected_to_vss() {
        let connections = diff_pair_physical_connections();
        assert_eq!(connections[4], ("S", "IBIAS"));
        assert_eq!(connections[5], ("B", "VSS"));
    }

    #[test]
    fn extrinsic_capacitances_form_a_passive_charge_conserving_matrix() {
        let matrix = combined_capacitance_matrix(
            MosCapacitanceMatrix {
                values: [[0.0; 4]; 4],
            },
            MosExtrinsicCapacitances {
                cgsol: 1.0,
                cgdol: 2.0,
                cjs: 3.0,
                cjd: 4.0,
            },
        );

        assert_eq!(
            matrix,
            [
                [3.0, -2.0, -1.0, 0.0],
                [-2.0, 6.0, 0.0, -4.0],
                [-1.0, 0.0, 4.0, -3.0],
                [0.0, -4.0, -3.0, 7.0],
            ]
        );
        for index in 0..4 {
            assert_eq!(matrix[index].iter().sum::<f64>(), 0.0);
            assert_eq!(matrix.iter().map(|row| row[index]).sum::<f64>(), 0.0);
        }
    }

    #[test]
    fn source_bulk_capacitance_cancels_when_both_terminals_share_a_node() {
        let matrix = combined_capacitance_matrix(
            MosCapacitanceMatrix {
                values: [[0.0; 4]; 4],
            },
            MosExtrinsicCapacitances {
                cgsol: 0.0,
                cgdol: 0.0,
                cjs: 3.0,
                cjd: 0.0,
            },
        );
        let collapsed = matrix[2][2] + matrix[2][3] + matrix[3][2] + matrix[3][3];
        assert_eq!(collapsed, 0.0);
    }

    #[test]
    fn ac_grid_has_twenty_points_per_decade_and_includes_endpoints() {
        let frequencies = ac_frequencies();
        let decades = (AC_MAX_HZ / AC_MIN_HZ).log10() as usize;
        assert_eq!(frequencies.len(), decades * AC_POINTS_PER_DECADE + 1);
        assert_eq!(frequencies[0], AC_MIN_HZ);
        assert!((frequencies[frequencies.len() - 1] - AC_MAX_HZ).abs() < 1.0e-3);
        assert!((frequencies[AC_POINTS_PER_DECADE] - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn normalized_laplace_variable_preserves_physical_s_c() {
        let frequency = 1.25e9;
        let capacitance = 2.5e-15;
        let normalized =
            laplace_frequency(frequency, AC_S_NORMALIZATION) * (capacitance * AC_S_NORMALIZATION);
        let expected = Complex::new(0.0, 2.0 * std::f64::consts::PI * frequency * capacitance);
        assert!((normalized.re - expected.re).abs() < 1.0e-30);
        assert!((normalized.im - expected.im).abs() < 1.0e-20);
    }

    #[test]
    fn crossing_is_interpolated_on_log_frequency() {
        let frequencies = [1.0, 10.0, 100.0];
        let values = [20.0, 10.0, -10.0];
        let (frequency, interval) = crossing_frequency(&frequencies, &values, 0.0).unwrap();
        assert_eq!(interval, 1);
        assert!((frequency - 10.0_f64.sqrt() * 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn gain_db_uses_the_gain_magnitude() {
        assert!(gain_db(1.0).abs() < 1.0e-12);
        assert!((gain_db(10.0) - 20.0).abs() < 1.0e-12);
        assert!((gain_db(-10.0) - 20.0).abs() < 1.0e-12);
    }

    #[test]
    fn dc_response_uses_low_frequency_limit_only_for_indeterminate_exact_dc() {
        let exact_dc = Complex::new(12.0, 0.0);
        let low_frequency = Complex::new(11.9, -0.1);
        assert_eq!(select_dc_response(exact_dc, low_frequency), exact_dc);

        let indeterminate_dc = Complex::new(f64::NAN, f64::NAN);
        assert_eq!(
            select_dc_response(indeterminate_dc, low_frequency),
            low_frequency
        );
    }

    #[test]
    fn ac_deltas_use_electrical_values_as_the_reference() {
        let bandwidth_change = relative_change_percent(Some(10.0e6), Some(8.0e6)).unwrap();
        assert!((bandwidth_change + 20.0).abs() < 1.0e-12);
        assert_eq!(relative_change_percent(None, Some(8.0e6)), None);
        assert_eq!(relative_change_percent(Some(0.0), Some(8.0e6)), None);
        assert_eq!(option_difference(Some(70.0), Some(62.5)), Some(-7.5));
        assert_eq!(option_difference(Some(70.0), None), None);
    }

    #[test]
    fn numeric_mna_solves_a_complex_linear_system() {
        let outputs = [
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
        ];

        let first = solve_numeric_mna(&outputs, 4, 2, 0, 1.0).unwrap().unwrap();
        let second = solve_numeric_mna(&outputs, 4, 2, 1, 1.0).unwrap().unwrap();

        assert!((first.re - 0.2).abs() < 1.0e-12);
        assert!(first.im.abs() < 1.0e-12);
        assert!((second.re - 0.6).abs() < 1.0e-12);
        assert!(second.im.abs() < 1.0e-12);
    }

    #[test]
    fn numeric_mna_reports_a_singular_system() {
        let outputs = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
        ];

        assert!(solve_numeric_mna(&outputs, 4, 2, 0, 1.0).unwrap().is_none());
    }

    #[test]
    fn ac_metrics_match_a_single_pole_response() {
        let pole_hz = 1.0e3;
        let frequencies = ac_frequencies();
        let responses = frequencies
            .iter()
            .map(|frequency| {
                let ratio = frequency / pole_hz;
                Complex::new(
                    -10.0 / (1.0 + ratio * ratio),
                    10.0 * ratio / (1.0 + ratio * ratio),
                )
            })
            .collect::<Vec<_>>();
        let metrics = ac_metrics_from_responses(&frequencies, &responses, Complex::new(-10.0, 0.0));

        assert!((metrics.dc_gain_db - 20.0).abs() < 1.0e-12);
        let bandwidth = metrics.bandwidth_3db_hz.unwrap();
        assert!((bandwidth / pole_hz - 1.0).abs() < 0.01);
        assert!(metrics.unity_gain_hz.is_some());
        let phase_margin = metrics.phase_margin_deg.unwrap();
        assert!(phase_margin > 90.0 && phase_margin < 100.0);
    }

    #[test]
    fn stage_counts_distinguish_failures_skips_and_exclusions() {
        let evaluations = [
            StageEvaluation::Evaluated {
                value: (),
                targets: AnalysisTargets::NONE.assess_dc_gain(0.0),
            },
            StageEvaluation::Evaluated {
                value: (),
                targets: ANALYSIS_TARGETS.assess_dc_gain(10.0),
            },
            StageEvaluation::Skipped {
                reason: SkipReason::DcTargets,
            },
            StageEvaluation::Excluded {
                reason: "outside LUT".to_owned(),
            },
        ];

        assert_eq!(
            count_stage(&evaluations),
            StageCounts {
                evaluated: 2,
                passed: 1,
                target_failed: 1,
                skipped: 1,
                excluded: 1,
            }
        );
    }

    #[test]
    fn target_status_reports_actual_and_unavailable_values() {
        let gain_target = ANALYSIS_TARGETS
            .min_dc_gain_db
            .expect("example configures a DC gain target");
        assert_eq!(
            format_target_assessment(&ANALYSIS_TARGETS.assess_dc_gain(19.0)),
            format!("fail: GainDC=19.000<{gain_target:.3}")
        );

        let metrics = AcMetrics {
            dc_gain_db: 20.0,
            bandwidth_3db_hz: Some(100.0e6),
            unity_gain_hz: None,
            phase_margin_deg: Some(45.0),
        };
        assert_eq!(
            format_target_assessment(&ANALYSIS_TARGETS.assess_frequency_metrics(&metrics)),
            "fail: UGF unavailable"
        );
    }
}
