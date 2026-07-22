use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use shapeic_lut::{DeviceLut, Expr, LookupTable, LutError, OperatingPoint};
use shapeic_mna::mna::{mna, mna_solve};
use symbolica::prelude::{Atom, AtomCore};

const TAIL_CURRENT: f64 = 20.0e-6;
const LENGTHS: [f64; 5] = [0.4e-6, 0.8e-6, 1.6e-6, 3.2e-6, 6.4e-6];
const VOUT_DC: f64 = 1.0;
const VG_DC: f64 = 0.9;
const SOURCE_VOLTAGE_START: f64 = 0.3;
const SOURCE_VOLTAGE_STOP: f64 = 0.8;
const SOURCE_VOLTAGE_POINTS: usize = 10;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";

type SmallSignalParameters = Vec<(String, f64)>;

struct SizedBlock {
    parameters: SmallSignalParameters,
    sizing: SizingSummary,
}

#[derive(Clone, Copy)]
struct SizingSummary {
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
    lut_sizing: Duration,
    numeric_mna: Duration,
    total: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let (nmos_path, pmos_path) = lut_paths()?;

    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let nmos_lut_load = stage_start.elapsed();
    let stage_start = Instant::now();
    let pmos_table = LookupTable::open(pmos_path)?;
    let pmos_lut_load = stage_start.elapsed();
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;
    let branch_current = TAIL_CURRENT / 2.0;

    let stage_start = Instant::now();
    let gain_expression = ota_gain_expression()?;
    let symbolic_mna = stage_start.elapsed();
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
    let mut diff_pairs = Vec::with_capacity(LENGTHS.len() * source_voltages.len());
    for length in LENGTHS {
        for (index, &source_voltage) in source_voltages.iter().enumerate() {
            let vgs = gate_source_voltages[index];
            let vds = drain_source_voltages[index];
            let block = simple_diff_pair(
                nmos,
                &OperatingPoint::new(length, 0.0, vgs, vds),
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
    let current_mirrors = LENGTHS
        .into_iter()
        .map(|length| {
            current_mirror(
                pmos,
                &OperatingPoint::new(length, 0.0, -VOUT_DC, -VOUT_DC),
                branch_current,
            )
            .map(|parameters| (length, parameters))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lut_sizing = stage_start.elapsed();
    let mut results = Vec::with_capacity(diff_pairs.len() * current_mirrors.len());

    let stage_start = Instant::now();
    for diff_point in &diff_pairs {
        for (mirror_length, current_mirror) in &current_mirrors {
            let parameters = diff_point
                .block
                .parameters
                .iter()
                .chain(&current_mirror.parameters)
                .cloned()
                .collect::<Vec<_>>();
            let gain = evaluate_ota_gain(&gain_expression, &parameters)?;
            results.push(SweepResult {
                diff_length: diff_point.length,
                mirror_length: *mirror_length,
                source_voltage: diff_point.source_voltage,
                diff_vgs: diff_point.vgs,
                diff_vds: diff_point.vds,
                diff_pair: diff_point.block.sizing,
                current_mirror: current_mirror.sizing,
                gain,
            });
        }
    }
    let numeric_mna = stage_start.elapsed();
    let timings = TimingSummary {
        nmos_lut_load,
        pmos_lut_load,
        symbolic_mna,
        lut_sizing,
        numeric_mna,
        total: total_start.elapsed(),
    };

    print_gain_table(&results);
    print_timing_table(&timings);
    Ok(())
}

fn simple_diff_pair(
    model: &DeviceLut,
    point: &OperatingPoint,
    branch_current: f64,
) -> Result<SizedBlock, LutError> {
    let expressions = [Expr::parameter("gm"), Expr::parameter("gds")];
    let sizing = model.size_for_current(point, branch_current, &expressions)?;
    let nf = f64::from(sizing.nf);
    let gm = sizing.values[0] * nf;
    let gds = sizing.values[1] * nf;

    Ok(SizedBlock {
        parameters: vec![
            ("g_gm_xdp".to_owned(), gm),
            ("r_gds_xdp".to_owned(), 1.0 / gds),
        ],
        sizing: sizing_summary(&sizing),
    })
}

fn current_mirror(
    model: &DeviceLut,
    point: &OperatingPoint,
    branch_current: f64,
) -> Result<SizedBlock, LutError> {
    let expressions = [Expr::parameter("gm"), Expr::parameter("gds")];
    let sizing = model.size_for_current(point, branch_current, &expressions)?;
    let nf = f64::from(sizing.nf);
    let gm = sizing.values[0] * nf;
    let gds = sizing.values[1] * nf;

    Ok(SizedBlock {
        parameters: vec![
            ("g_gm_xcm".to_owned(), gm),
            ("r_gds_xcm".to_owned(), 1.0 / gds),
        ],
        sizing: sizing_summary(&sizing),
    })
}

fn sizing_summary(sizing: &shapeic_lut::CurrentSizingResult) -> SizingSummary {
    SizingSummary {
        finger_width: sizing.point.finger_width,
        nf: sizing.nf,
        requested_current: sizing.requested_current,
        predicted_current: sizing.predicted_current,
        current_error: sizing.current_error,
    }
}

fn ota_gain_expression() -> Result<Atom, io::Error> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spice_dir = manifest.join("examples/ota_4t");
    let output_dir = manifest.join("../target/shapeic-ota-4t");
    let system = mna(&spice_dir, &output_dir, "ota_4t")
        .map_err(|error| io::Error::other(format!("could not build OTA MNA: {error:?}")))?;
    let output_node = system
        .nodes
        .nodes
        .get("VOUT")
        .copied()
        .ok_or_else(|| io::Error::other("OTA netlist has no VOUT node"))?;
    let output_row = output_node
        .checked_sub(1)
        .ok_or_else(|| io::Error::other("VOUT cannot be the ground node"))?;
    let output_row = u32::try_from(output_row)
        .map_err(|_| io::Error::other("VOUT row does not fit the MNA matrix index"))?;
    let solution = mna_solve(&system.a, &system.x, &system.z)
        .map_err(|error| io::Error::other(format!("could not solve OTA MNA: {error:?}")))?;
    Ok(solution.solution[(output_row, 0)].clone())
}

fn evaluate_ota_gain(
    gain_expression: &Atom,
    parameters: &[(String, f64)],
) -> Result<f64, io::Error> {
    let symbols = gain_expression
        .get_all_symbols(false)
        .into_iter()
        .map(Atom::from)
        .collect::<Vec<_>>();
    let values = symbols
        .iter()
        .map(|symbol| {
            let name = symbol.to_string();
            parameters
                .iter()
                .find_map(|(parameter, value)| (parameter == &name).then_some(*value))
                .ok_or_else(|| io::Error::other(format!("no value for MNA parameter '{name}'")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let evaluator = gain_expression
        .evaluator(&symbols)
        .build()
        .map_err(|error| io::Error::other(format!("could not evaluate OTA gain: {error}")))?;
    let mut evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
    let gain = evaluator.evaluate_single(&values);
    if !gain.is_finite() {
        return Err(io::Error::other("OTA gain is not finite"));
    }
    Ok(gain)
}

fn lut_paths() -> Result<(PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments.next().unwrap_or_else(|| OsString::from("ota_4t"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz>",
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

fn print_gain_table(results: &[SweepResult]) {
    println!(
        "{:>9} | {:>9} | {:>7} | {:>9} | {:>9} | {:>10} | {:>10} | {:>5} | {:>10} | {:>10} | {:>10} | {:>5} | {:>10} | {:>10} | {:>16}",
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
    );
    println!(
        "----------+-----------+---------+-----------+-----------+------------+------------+-------+------------+------------+------------+-------+------------+------------+-----------------"
    );
    for result in results {
        println!(
            "{:>9.3} | {:>9.3} | {:>7.3} | {:>9.3} | {:>9.3} | {:>10.6} | {:>10.6} | {:>5} | {:>10.6} | {:>10.6} | {:>10.6} | {:>5} | {:>10.6} | {:>10.6} | {:>16.6e}",
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
        );
    }
}

fn print_timing_table(timings: &TimingSummary) {
    println!();
    println!("{:<28} | {:>12}", "Stage", "Time [ms]");
    println!("-----------------------------+-------------");
    for (stage, duration) in [
        ("NMOS LUT load", timings.nmos_lut_load),
        ("PMOS LUT load", timings.pmos_lut_load),
        ("Symbolic MNA build + solve", timings.symbolic_mna),
        ("LUT sizing + interpolation", timings.lut_sizing),
        ("Numeric MNA evaluation", timings.numeric_mna),
        ("Total computation", timings.total),
    ] {
        println!("{stage:<28} | {:>12.3}", duration.as_secs_f64() * 1.0e3);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SOURCE_VOLTAGE_POINTS, SOURCE_VOLTAGE_START, SOURCE_VOLTAGE_STOP, VG_DC, VOUT_DC, linspace,
        normalize_voltage,
    };

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
        assert_eq!(drain_source_voltages[0], 0.7);
        assert_eq!(drain_source_voltages[SOURCE_VOLTAGE_POINTS - 1], 0.1);
        assert_eq!(gate_source_voltages[0], 0.7);
        assert_eq!(gate_source_voltages[SOURCE_VOLTAGE_POINTS - 1], 0.1);
    }
}
