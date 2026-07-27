use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use num_complex::Complex64;
pub use shapeic_core::analysis::AcMetrics;
use shapeic_core::analysis::{
    AdaptiveAcConfig, AdaptiveAcOutcome, AdaptiveAcPolicy, analyze_adaptive_ac,
};
use shapeic_layout::{PhysicalLookupTable, PhysicalPoint, PortAdmittance};
use shapeic_lut::{
    CurrentSizingResult, DeviceLut, Expr, LutError, MosCapacitanceMatrix, MosExtrinsicCapacitances,
    OperatingPoint,
};
use shapeic_mna::mna::{MnaResult, mna, stamp_port_admittance};
use shapeic_mna::numeric::{NumericMnaSystem, PreparedNumericMna};
use symbolica::domains::float::Complex;
use symbolica::prelude::{Atom, AtomCore, ExpressionEvaluator, Matrix};

pub const AC_MIN_HZ: f64 = 1.0;
pub const AC_MAX_HZ: f64 = 100.0e9;
pub const AC_POINTS_PER_DECADE: usize = 20;
pub const ADAPTIVE_COARSE_POINTS_PER_DECADE: usize = 4;
pub const ADAPTIVE_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
pub const ADAPTIVE_MAX_REFINEMENT_STEPS: usize = 32;

const AC_S_NORMALIZATION: f64 = 1.0e12;
const MNA_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];

type SmallSignalParameters = Vec<(String, f64)>;

struct SizedBlock {
    parameters: SmallSignalParameters,
    sizing: SizingSummary,
    intrinsic_capacitance: MosCapacitanceMatrix,
    extrinsic_capacitance: MosExtrinsicCapacitances,
}

#[derive(Clone, Copy, Debug)]
pub struct SizingSummary {
    pub length: f64,
    pub finger_width: f64,
    pub nf: u32,
    pub total_width: f64,
    pub requested_current: f64,
    pub predicted_current: f64,
    pub current_error: f64,
}

impl SizingSummary {
    pub fn relative_error_percent(self) -> f64 {
        100.0 * self.current_error / self.requested_current
    }
}

#[derive(Clone, Copy, Debug)]
struct AcSample {
    frequency_hz: f64,
    response: Complex<f64>,
    gain_db: f64,
    phase_deg: f64,
}

#[derive(Clone, Debug)]
pub struct AcSweep {
    samples: Vec<AcSample>,
}

impl AcSweep {
    pub fn write_csv(&self, path: &Path) -> Result<(), io::Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(
            writer,
            "frequency_hz,response_real,response_imag,gain_db,phase_deg"
        )?;
        for sample in &self.samples {
            writeln!(
                writer,
                "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                sample.frequency_hz,
                sample.response.re,
                sample.response.im,
                sample.gain_db,
                sample.phase_deg
            )?;
        }
        writer.flush()
    }
}

#[derive(Clone, Debug)]
pub struct ElectricalAnalysis {
    pub diff_pair: SizingSummary,
    pub current_mirror: SizingSummary,
    pub metrics: AcMetrics,
    pub ac_sweep: AcSweep,
}

#[derive(Clone, Debug)]
pub struct ElectricalAdaptiveComparison {
    pub diff_pair: SizingSummary,
    pub current_mirror: SizingSummary,
    pub dense_metrics: AcMetrics,
    pub dense_sweep: AcSweep,
    pub adaptive_metrics: AcMetrics,
    pub adaptive_sweep: AcSweep,
    pub adaptive_frequency_evaluations: usize,
}

#[derive(Clone, Debug)]
pub struct PhysicalAnalysis {
    pub diff_pair: SizingSummary,
    pub current_mirror: SizingSummary,
    pub metrics: AcMetrics,
    pub ac_sweep: AcSweep,
}

struct EvaluatedAc {
    metrics: AcMetrics,
    sweep: AcSweep,
}

pub fn analyze(
    nmos: &DeviceLut,
    pmos: &DeviceLut,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
    branch_current: f64,
    spice_dir: &Path,
    mna_output_dir: &Path,
) -> Result<ElectricalAnalysis, Box<dyn Error>> {
    validate_capacitance_parameters(nmos)?;
    validate_capacitance_parameters(pmos)?;

    let diff_pair = size_block(nmos, &diff_point, branch_current, "g_gm_xdp", "r_gds_xdp")?;
    let current_mirror = size_block(pmos, &mirror_point, branch_current, "g_gm_xcm", "r_gds_xcm")?;
    let system = mna(spice_dir, mna_output_dir, "ota_4t")
        .map_err(|error| io::Error::other(format!("could not build OTA MNA: {error:?}")))?;
    let ac = evaluate_electrical_ac(&system, &diff_pair, &current_mirror, "IBIAS")?;

    Ok(ElectricalAnalysis {
        diff_pair: diff_pair.sizing,
        current_mirror: current_mirror.sizing,
        metrics: ac.metrics,
        ac_sweep: ac.sweep,
    })
}

pub fn analyze_adaptive_comparison(
    nmos: &DeviceLut,
    pmos: &DeviceLut,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
    branch_current: f64,
    spice_dir: &Path,
    mna_output_dir: &Path,
) -> Result<ElectricalAdaptiveComparison, Box<dyn Error>> {
    validate_capacitance_parameters(nmos)?;
    validate_capacitance_parameters(pmos)?;

    let diff_pair = size_block(nmos, &diff_point, branch_current, "g_gm_xdp", "r_gds_xdp")?;
    let current_mirror = size_block(pmos, &mirror_point, branch_current, "g_gm_xcm", "r_gds_xcm")?;
    let system = mna(spice_dir, mna_output_dir, "ota_4t")
        .map_err(|error| io::Error::other(format!("could not build OTA MNA: {error:?}")))?;
    let numeric = electrical_numeric_mna(&system, &diff_pair, &current_mirror, "IBIAS")?;
    let dense = evaluate_dense_numeric_ac(&numeric)?;
    let adaptive = evaluate_adaptive_numeric_ac(&numeric, true)?;
    let adaptive_sweep = adaptive_sweep(&adaptive);

    Ok(ElectricalAdaptiveComparison {
        diff_pair: diff_pair.sizing,
        current_mirror: current_mirror.sizing,
        dense_metrics: dense.metrics,
        dense_sweep: dense.sweep,
        adaptive_metrics: adaptive.metrics,
        adaptive_sweep,
        adaptive_frequency_evaluations: adaptive.frequency_evaluations,
    })
}

pub fn analyze_physical(
    nmos: &DeviceLut,
    pmos: &DeviceLut,
    physical: &PhysicalLookupTable,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
    branch_current: f64,
    spice_dir: &Path,
    mna_output_dir: &Path,
) -> Result<PhysicalAnalysis, Box<dyn Error>> {
    validate_capacitance_parameters(nmos)?;
    validate_capacitance_parameters(pmos)?;

    let diff_pair = size_block(nmos, &diff_point, branch_current, "g_gm_xdp", "r_gds_xdp")?;
    let current_mirror = size_block(pmos, &mirror_point, branch_current, "g_gm_xcm", "r_gds_xcm")?;
    let diff_physical = query_physical(physical, "simplediffpair", diff_pair.sizing)?;
    let mirror_physical = query_physical(physical, "currentmirror", current_mirror.sizing)?;
    let system = mna(spice_dir, mna_output_dir, "ota_4t")
        .map_err(|error| io::Error::other(format!("could not build OTA MNA: {error:?}")))?;
    let ac = evaluate_physical_ac(
        &system,
        &diff_pair,
        &current_mirror,
        &diff_physical,
        &mirror_physical,
    )?;

    Ok(PhysicalAnalysis {
        diff_pair: diff_pair.sizing,
        current_mirror: current_mirror.sizing,
        metrics: ac.metrics,
        ac_sweep: ac.sweep,
    })
}

fn size_block(
    model: &DeviceLut,
    point: &OperatingPoint,
    branch_current: f64,
    gm_parameter: &str,
    gds_resistance_parameter: &str,
) -> Result<SizedBlock, LutError> {
    let sizing = model.size_for_current(point, branch_current, &sizing_expressions())?;
    let nf = f64::from(sizing.nf);
    let gm = sizing.values[0] * nf;
    let gds = sizing.values[1] * nf;
    let intrinsic_capacitance = MosCapacitanceMatrix::from_flat(&sizing.values[2..])?.scaled(nf);
    let extrinsic_capacitance = sizing.extrinsic_capacitances.ok_or_else(|| {
        LutError::ExtrinsicCapacitanceSamplesUnavailable {
            model: model.name().to_owned(),
        }
    })?;

    Ok(SizedBlock {
        parameters: vec![
            (gm_parameter.to_owned(), gm),
            (gds_resistance_parameter.to_owned(), 1.0 / gds),
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

fn sizing_summary(sizing: &CurrentSizingResult) -> SizingSummary {
    SizingSummary {
        length: sizing.point.operating_point.length,
        finger_width: sizing.point.finger_width,
        nf: sizing.nf,
        total_width: sizing.total_width,
        requested_current: sizing.requested_current,
        predicted_current: sizing.predicted_current,
        current_error: sizing.current_error,
    }
}

fn validate_capacitance_parameters(model: &DeviceLut) -> Result<(), io::Error> {
    let missing = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
        .into_iter()
        .filter(|required| {
            !model
                .parameter_names()
                .iter()
                .any(|parameter| parameter == required)
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    Err(io::Error::other(format!(
        "model '{}' is missing intrinsic capacitances: {}",
        model.name(),
        missing.join(", ")
    )))
}

fn evaluate_electrical_ac(
    system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    diff_bulk_node: &str,
) -> Result<EvaluatedAc, io::Error> {
    let numeric = electrical_numeric_mna(system, diff_pair, current_mirror, diff_bulk_node)?;
    evaluate_dense_numeric_ac(&numeric)
}

fn electrical_numeric_mna(
    system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    diff_bulk_node: &str,
) -> Result<NumericMnaSystem, io::Error> {
    let mut prepared = PreparedNumericMna::new(system, &MNA_PARAMETER_ORDER).map_err(|error| {
        io::Error::other(format!(
            "could not prepare numerical electrical MNA: {error}"
        ))
    })?;
    let values = ota_parameter_values(diff_pair, current_mirror)?;
    let mut numeric = prepared.instantiate(&values).map_err(|error| {
        io::Error::other(format!(
            "could not instantiate numerical electrical MNA: {error}"
        ))
    })?;
    stamp_numeric_device_capacitances(&mut numeric, diff_pair, current_mirror, diff_bulk_node)?;
    Ok(numeric)
}

fn evaluate_physical_ac(
    system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    diff_physical: &PortAdmittance,
    mirror_physical: &PortAdmittance,
) -> Result<EvaluatedAc, io::Error> {
    let mut matrix = system.a.clone();
    stamp_device_capacitances(&mut matrix, system, diff_pair, current_mirror, "VSS")?;
    stamp_physical_admittance(
        &mut matrix,
        system,
        diff_physical,
        [
            ("DP", "VOUT"),
            ("DN", "N1"),
            ("GP", "VINP"),
            ("GN", "VINN"),
            ("S", "IBIAS"),
            ("B", "VSS"),
        ],
    )?;
    stamp_physical_admittance(
        &mut matrix,
        system,
        mirror_physical,
        [("DOUT", "VOUT"), ("DREF", "N1"), ("S", "VDD"), ("B", "VDD")],
    )?;
    let parameters = diff_pair
        .parameters
        .iter()
        .chain(&current_mirror.parameters)
        .cloned()
        .collect::<Vec<_>>();
    evaluate_numeric_ac(&matrix, system, &parameters)
}

fn ota_parameter_values(
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
) -> Result<[f64; 4], io::Error> {
    let parameter = |block: &SizedBlock, name: &str| {
        block
            .parameters
            .iter()
            .find_map(|(parameter, value)| (parameter == name).then_some(*value))
            .ok_or_else(|| io::Error::other(format!("no value for MNA parameter '{name}'")))
    };
    Ok([
        parameter(diff_pair, MNA_PARAMETER_ORDER[0])?,
        parameter(diff_pair, MNA_PARAMETER_ORDER[1])?,
        parameter(current_mirror, MNA_PARAMETER_ORDER[2])?,
        parameter(current_mirror, MNA_PARAMETER_ORDER[3])?,
    ])
}

fn query_physical(
    table: &PhysicalLookupTable,
    primitive: &str,
    sizing: SizingSummary,
) -> Result<PortAdmittance, Box<dyn Error>> {
    Ok(table.primitive(primitive)?.query(PhysicalPoint::new(
        sizing.length,
        sizing.finger_width,
        sizing.nf,
    ))?)
}

fn diff_pair_connections<'a>(
    gate: &'a str,
    drain: &'a str,
    bulk: &'a str,
) -> [(&'static str, &'a str); 4] {
    [("G", gate), ("D", drain), ("S", "IBIAS"), ("B", bulk)]
}

fn stamp_device_capacitances(
    matrix: &mut Matrix<symbolica::domains::atom::AtomField>,
    system: &MnaResult,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    diff_bulk_node: &str,
) -> Result<(), io::Error> {
    stamp_device_capacitance(
        matrix,
        system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_connections("VINP", "VOUT", diff_bulk_node),
    )?;
    stamp_device_capacitance(
        matrix,
        system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_connections("VINN", "N1", diff_bulk_node),
    )?;
    stamp_device_capacitance(
        matrix,
        system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD")],
    )?;
    stamp_device_capacitance(
        matrix,
        system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD")],
    )
}

fn stamp_numeric_device_capacitances(
    system: &mut NumericMnaSystem,
    diff_pair: &SizedBlock,
    current_mirror: &SizedBlock,
    diff_bulk_node: &str,
) -> Result<(), io::Error> {
    stamp_numeric_device_capacitance(
        system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_connections("VINP", "VOUT", diff_bulk_node),
    )?;
    stamp_numeric_device_capacitance(
        system,
        diff_pair.intrinsic_capacitance,
        diff_pair.extrinsic_capacitance,
        diff_pair_connections("VINN", "N1", diff_bulk_node),
    )?;
    stamp_numeric_device_capacitance(
        system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "VOUT"), ("S", "VDD"), ("B", "VDD")],
    )?;
    stamp_numeric_device_capacitance(
        system,
        current_mirror.intrinsic_capacitance,
        current_mirror.extrinsic_capacitance,
        [("G", "N1"), ("D", "N1"), ("S", "VDD"), ("B", "VDD")],
    )
}

fn stamp_physical_admittance<const N: usize>(
    matrix: &mut Matrix<symbolica::domains::atom::AtomField>,
    system: &MnaResult,
    admittance: &PortAdmittance,
    connections: [(&str, &str); N],
) -> Result<(), io::Error> {
    let connections = connections
        .into_iter()
        .map(|(port, node)| (port.to_owned(), node.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let capacitance = admittance
        .capacitance
        .mapv(|value| value * AC_S_NORMALIZATION);
    stamp_port_admittance(
        matrix,
        &system.nodes,
        &admittance.ports,
        &connections,
        admittance.conductance.view(),
        capacitance.view(),
    )
    .map_err(|error| {
        io::Error::other(format!(
            "could not stamp physical primitive admittance: {error:?}"
        ))
    })
}

fn stamp_device_capacitance<const N: usize>(
    matrix: &mut Matrix<symbolica::domains::atom::AtomField>,
    system: &MnaResult,
    intrinsic: MosCapacitanceMatrix,
    extrinsic: MosExtrinsicCapacitances,
    connections: [(&str, &str); N],
) -> Result<(), io::Error> {
    let capacitance = combined_capacitance_matrix(intrinsic, extrinsic);
    let flat = capacitance.into_iter().flatten().collect::<Vec<_>>();
    let capacitance = Array2::from_shape_vec((4, 4), flat)
        .expect("fixed 4x4 capacitance matrix")
        .mapv(|value| value * AC_S_NORMALIZATION);
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
    .map_err(|error| io::Error::other(format!("could not stamp MOS capacitance: {error:?}")))
}

fn stamp_numeric_device_capacitance<const N: usize>(
    system: &mut NumericMnaSystem,
    intrinsic: MosCapacitanceMatrix,
    extrinsic: MosExtrinsicCapacitances,
    connections: [(&str, &str); N],
) -> Result<(), io::Error> {
    let flat = combined_capacitance_matrix(intrinsic, extrinsic)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let capacitance = Array2::from_shape_vec((4, 4), flat).expect("fixed 4x4 capacitance matrix");
    let conductance = Array2::<f64>::zeros((4, 4));
    let ports = MosCapacitanceMatrix::TERMINALS.map(str::to_owned).to_vec();
    let connections = connections
        .into_iter()
        .map(|(port, node)| (port.to_owned(), node.to_owned()))
        .collect::<BTreeMap<_, _>>();
    system
        .stamp_port_admittance(&ports, &connections, conductance.view(), capacitance.view())
        .map_err(|error| {
            io::Error::other(format!(
                "could not stamp numerical MOS capacitance: {error}"
            ))
        })
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

fn evaluate_dense_numeric_ac(system: &NumericMnaSystem) -> Result<EvaluatedAc, io::Error> {
    let frequencies = ac_frequencies();
    let responses = frequencies
        .iter()
        .map(|frequency| {
            system
                .solve_node(*frequency, "VOUT")
                .map_err(|error| io::Error::other(error.to_string()))?
                .map(|response| Complex::new(response.re, response.im))
                .ok_or_else(|| {
                    io::Error::other(format!("numeric OTA MNA is singular at {frequency:.6e} Hz"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dc_response = system
        .solve_node(0.0, "VOUT")
        .map_err(|error| io::Error::other(error.to_string()))?
        .map(|response| Complex::new(response.re, response.im))
        .unwrap_or(responses[0]);
    Ok(ac_from_responses(&frequencies, &responses, dc_response))
}

fn evaluate_adaptive_numeric_ac(
    system: &NumericMnaSystem,
    retain_samples: bool,
) -> Result<AdaptiveAcOutcome, io::Error> {
    let config = AdaptiveAcConfig {
        min_frequency_hz: AC_MIN_HZ,
        max_frequency_hz: AC_MAX_HZ,
        coarse_points_per_decade: ADAPTIVE_COARSE_POINTS_PER_DECADE,
        crossing_relative_tolerance: ADAPTIVE_CROSSING_RELATIVE_TOLERANCE,
        max_refinement_steps: ADAPTIVE_MAX_REFINEMENT_STEPS,
        retain_samples,
    };
    analyze_adaptive_ac(config, AdaptiveAcPolicy::COMPLETE, |frequency| match system
        .solve_node(frequency, "VOUT")
        .map_err(|error| io::Error::other(error.to_string()))?
    {
        Some(response) => Ok(-response),
        None if frequency == 0.0 => Ok(Complex64::new(f64::NAN, f64::NAN)),
        None => Err(io::Error::other(format!(
            "numeric OTA MNA is singular at {frequency:.6e} Hz"
        ))),
    })
    .map_err(|error| io::Error::other(format!("adaptive electrical AC failed: {error}")))
}

fn evaluate_numeric_ac(
    matrix: &Matrix<symbolica::domains::atom::AtomField>,
    system: &MnaResult,
    parameters: &[(String, f64)],
) -> Result<EvaluatedAc, io::Error> {
    let mut evaluator = NumericAcEvaluator::new(matrix, system, parameters)?;
    evaluate_dense_ac(&mut evaluator)
}

struct NumericAcEvaluator {
    symbols: Vec<Atom>,
    evaluator: ExpressionEvaluator<Complex<f64>>,
    parameter_map: BTreeMap<String, f64>,
    outputs: Vec<Complex<f64>>,
    matrix_len: usize,
    rows: usize,
    output_row: usize,
}

impl NumericAcEvaluator {
    fn new(
        matrix: &Matrix<symbolica::domains::atom::AtomField>,
        system: &MnaResult,
        parameters: &[(String, f64)],
    ) -> Result<Self, io::Error> {
        let rows = matrix.nrows();
        let columns = matrix.ncols();
        if rows != columns || system.z.nrows() != rows || system.z.ncols() != 1 {
            return Err(io::Error::other(
                "numeric AC analysis requires a square MNA and one matching RHS column",
            ));
        }

        let matrix_len = rows * columns;
        let mut expressions = Vec::with_capacity(matrix_len + rows);
        for row in 0..rows {
            let row = u32::try_from(row)
                .map_err(|_| io::Error::other("OTA MNA row index exceeds u32"))?;
            for column in 0..columns {
                let column = u32::try_from(column)
                    .map_err(|_| io::Error::other("OTA MNA column index exceeds u32"))?;
                expressions.push(matrix[(row, column)].clone());
            }
        }
        for row in 0..rows {
            let row = u32::try_from(row)
                .map_err(|_| io::Error::other("OTA MNA row index exceeds u32"))?;
            expressions.push(system.z[(row, 0)].clone());
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
            })?
            .map_coeff(&|coefficient| {
                Complex::new(coefficient.re.to_f64(), coefficient.im.to_f64())
            });
        Ok(Self {
            symbols,
            evaluator,
            parameter_map: parameters.iter().cloned().collect(),
            outputs: vec![Complex::new(0.0, 0.0); expressions.len()],
            matrix_len,
            rows,
            output_row: ota_output_row(system)?,
        })
    }

    fn evaluate(&mut self, frequency_hz: f64) -> Result<Complex<f64>, io::Error> {
        self.evaluate_optional(frequency_hz)?.ok_or_else(|| {
            io::Error::other(format!(
                "numeric OTA MNA is singular at {frequency_hz:.6e} Hz"
            ))
        })
    }

    fn evaluate_optional(&mut self, frequency_hz: f64) -> Result<Option<Complex<f64>>, io::Error> {
        let values = evaluator_inputs(&self.symbols, &self.parameter_map, frequency_hz)?;
        self.evaluator
            .try_evaluate(&values, &mut self.outputs)
            .map_err(|error| {
                io::Error::other(format!(
                    "could not evaluate numeric MNA at {frequency_hz:.6e} Hz: {error}"
                ))
            })?;
        solve_numeric_mna(
            &self.outputs,
            self.matrix_len,
            self.rows,
            self.output_row,
            frequency_hz,
        )
    }
}

fn evaluate_dense_ac(evaluator: &mut NumericAcEvaluator) -> Result<EvaluatedAc, io::Error> {
    let frequencies = ac_frequencies();
    let responses = frequencies
        .iter()
        .map(|frequency| evaluator.evaluate(*frequency))
        .collect::<Result<Vec<_>, _>>()?;
    let dc_response = evaluator.evaluate_optional(0.0)?.unwrap_or(responses[0]);
    Ok(ac_from_responses(&frequencies, &responses, dc_response))
}

fn adaptive_sweep(outcome: &AdaptiveAcOutcome) -> AcSweep {
    let samples = outcome
        .samples
        .iter()
        .map(|sample| AcSample {
            frequency_hz: sample.frequency_hz,
            response: Complex::new(sample.response.re, sample.response.im),
            gain_db: sample.gain_db,
            phase_deg: sample.phase_deg,
        })
        .collect();
    AcSweep { samples }
}

fn evaluator_inputs(
    symbols: &[Atom],
    parameter_map: &BTreeMap<String, f64>,
    frequency_hz: f64,
) -> Result<Vec<Complex<f64>>, io::Error> {
    symbols
        .iter()
        .map(|symbol| {
            let name = symbol.to_string();
            if name == "s" {
                Ok(Complex::new(
                    0.0,
                    2.0 * std::f64::consts::PI * frequency_hz / AC_S_NORMALIZATION,
                ))
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

fn solve_numeric_mna(
    values: &[Complex<f64>],
    matrix_len: usize,
    rows: usize,
    output_row: usize,
    frequency_hz: f64,
) -> Result<Option<Complex<f64>>, io::Error> {
    let expected_matrix_len = rows
        .checked_mul(rows)
        .ok_or_else(|| io::Error::other("numeric MNA dimensions overflow usize"))?;
    let expected_values = expected_matrix_len
        .checked_add(rows)
        .ok_or_else(|| io::Error::other("numeric MNA output count overflows usize"))?;
    if rows == 0
        || matrix_len != expected_matrix_len
        || values.len() != expected_values
        || output_row >= rows
    {
        return Err(io::Error::other("invalid numeric OTA MNA dimensions"));
    }
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(io::Error::other(format!(
            "numeric OTA MNA contains a non-finite value at {frequency_hz:.6e} Hz"
        )));
    }

    let coefficients = values[..matrix_len]
        .iter()
        .map(|value| Complex64::new(value.re, value.im))
        .collect::<Vec<_>>();
    let rhs = values[matrix_len..]
        .iter()
        .map(|value| Complex64::new(value.re, value.im))
        .collect::<Vec<_>>();
    let Some(solution) = DMatrix::from_row_slice(rows, rows, &coefficients)
        .lu()
        .solve(&DVector::from_vec(rhs))
    else {
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

fn ota_output_row(system: &MnaResult) -> Result<usize, io::Error> {
    let node = system
        .nodes
        .nodes
        .get("VOUT")
        .copied()
        .ok_or_else(|| io::Error::other("OTA MNA has no VOUT node"))?;
    node.checked_sub(1)
        .ok_or_else(|| io::Error::other("VOUT cannot be ground"))
}

fn ac_frequencies() -> Vec<f64> {
    let decades = (AC_MAX_HZ / AC_MIN_HZ).log10();
    let count = (decades * AC_POINTS_PER_DECADE as f64).round() as usize + 1;
    (0..count)
        .map(|index| AC_MIN_HZ * 10.0_f64.powf(index as f64 / AC_POINTS_PER_DECADE as f64))
        .collect()
}

fn ac_from_responses(
    frequencies: &[f64],
    amplifier_responses: &[Complex<f64>],
    amplifier_dc_response: Complex<f64>,
) -> EvaluatedAc {
    debug_assert_eq!(frequencies.len(), amplifier_responses.len());
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
    let samples = frequencies
        .iter()
        .copied()
        .zip(loop_responses.iter().copied())
        .zip(magnitudes.iter().copied())
        .zip(phases.iter().copied())
        .map(
            |(((frequency_hz, response), gain_db), phase_deg)| AcSample {
                frequency_hz,
                response,
                gain_db,
                phase_deg,
            },
        )
        .collect();
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

    EvaluatedAc {
        metrics: AcMetrics {
            dc_gain_db,
            bandwidth_3db_hz,
            unity_gain_hz,
            phase_margin_deg,
        },
        sweep: AcSweep { samples },
    }
}

fn loop_response(amplifier_response: Complex<f64>) -> Complex<f64> {
    -amplifier_response
}

fn magnitude_db(value: Complex<f64>) -> f64 {
    10.0 * (value.re * value.re + value.im * value.im).log10()
}

fn unwrap_phases(responses: &[Complex<f64>]) -> Vec<f64> {
    let mut phases = Vec::with_capacity(responses.len());
    for response in responses {
        let mut phase = response.im.atan2(response.re).to_degrees();
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn ac_grid_matches_the_ota_example() {
        let frequencies = ac_frequencies();
        assert_eq!(frequencies.len(), 221);
        assert_eq!(frequencies[0], AC_MIN_HZ);
        assert!((frequencies[20] - 10.0).abs() < 1.0e-12);
        assert!((frequencies[220] - AC_MAX_HZ).abs() < 1.0e-3);
    }

    #[test]
    fn diff_pair_bulk_tracks_the_source_node() {
        let connections = diff_pair_connections("VINP", "VOUT", "IBIAS");
        assert_eq!(connections[2], ("S", "IBIAS"));
        assert_eq!(connections[3], ("B", "IBIAS"));
    }

    #[test]
    fn physical_diff_pair_bulk_is_connected_to_vss() {
        let connections = diff_pair_connections("VINP", "VOUT", "VSS");
        assert_eq!(connections[2], ("S", "IBIAS"));
        assert_eq!(connections[3], ("B", "VSS"));
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
    fn metrics_find_a_single_pole_bandwidth() {
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
        let ac = ac_from_responses(&frequencies, &responses, Complex::new(-10.0, 0.0));
        let metrics = ac.metrics;

        assert!((metrics.dc_gain_db - 20.0).abs() < 1.0e-12);
        assert!((metrics.bandwidth_3db_hz.unwrap() / pole_hz - 1.0).abs() < 0.01);
        assert!(metrics.unity_gain_hz.is_some());
        let phase_margin = metrics.phase_margin_deg.unwrap();
        assert!(phase_margin > 90.0 && phase_margin < 100.0);
        assert_eq!(ac.sweep.samples.len(), frequencies.len());
        assert_eq!(ac.sweep.samples[0].frequency_hz, frequencies[0]);
        assert!((ac.sweep.samples[0].gain_db - magnitude_db(responses[0])).abs() < 1.0e-12);
        assert_eq!(ac.sweep.samples[0].response, loop_response(responses[0]));
    }

    #[test]
    fn writes_the_same_samples_used_for_ac_metrics() {
        let frequencies = [1.0, 10.0];
        let responses = [Complex::new(2.0, -1.0), Complex::new(1.0, -2.0)];
        let ac = ac_from_responses(&frequencies, &responses, responses[0]);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "shapeic-ota-ac-{}-{timestamp}.csv",
            std::process::id()
        ));

        ac.sweep.write_csv(&path).expect("write AC sweep");
        let text = fs::read_to_string(&path).expect("read AC sweep");
        fs::remove_file(path).expect("cleanup");
        let lines = text.lines().collect::<Vec<_>>();

        assert_eq!(
            lines[0],
            "frequency_hz,response_real,response_imag,gain_db,phase_deg"
        );
        assert_eq!(lines.len(), frequencies.len() + 1);
        let first = lines[1]
            .split(',')
            .map(|value| value.parse::<f64>().expect("numeric CSV value"))
            .collect::<Vec<_>>();
        let loop_responses = responses.map(loop_response);
        assert_eq!(first[0], frequencies[0]);
        assert_eq!(first[1], loop_responses[0].re);
        assert_eq!(first[2], loop_responses[0].im);
        assert!((first[3] - magnitude_db(loop_responses[0])).abs() < 1.0e-12);
        assert!((first[4] - unwrap_phases(&loop_responses)[0]).abs() < 1.0e-12);
    }

    #[test]
    fn inverting_amplifier_response_becomes_zero_referenced_loop_phase() {
        let responses = [
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0, 0.1),
            Complex::new(0.0, 1.0),
        ];
        let loop_responses = responses.map(loop_response);
        let phases = unwrap_phases(&loop_responses);
        assert!(phases[0].abs() < 1.0e-12);
        assert!(phases[1] < 0.0);
        assert!((phases[2] + 90.0).abs() < 1.0e-12);
    }
}
