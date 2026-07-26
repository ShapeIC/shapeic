#[allow(dead_code)]
#[path = "ota_4t_electrical_verification/analysis.rs"]
mod analysis;
#[path = "ota_4t_electrical_verification/common.rs"]
mod common;

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufWriter, Write as _};
use std::path::{Path, PathBuf};

use analysis::{AcMetrics, ElectricalAdaptiveComparison, SizingSummary};
use common::{
    NMOS_MODEL, PMOS_MODEL, TAIL_CURRENT, VDD_DC, VG_DC, VOUT_DC, unique_output_root,
    validate_finger_width, verification_config,
};
use shapeic_lut::verification::{VerificationEngine, VerificationInput};
use shapeic_lut::{LookupTable, OperatingPoint};

const DIFF_LENGTH: f64 = 1.6e-6;
const MIRROR_LENGTH: f64 = 6.4e-6;
const SOURCE_VOLTAGE: f64 = 0.672_822_822_822_822_8;
const METRICS: [&str; 4] = [
    "dc_gain_db",
    "bandwidth_3db_hz",
    "unity_gain_hz",
    "phase_margin_deg",
];

fn main() -> Result<(), Box<dyn Error>> {
    let (nmos_path, pmos_path) = lut_paths()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_root =
        unique_output_root(manifest, "shapeic-ota-4t-electrical-adaptive-verification")?;
    let diff_point = OperatingPoint::new(
        DIFF_LENGTH,
        0.0,
        VG_DC - SOURCE_VOLTAGE,
        VOUT_DC - SOURCE_VOLTAGE,
    );
    let mirror_point = OperatingPoint::new(MIRROR_LENGTH, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC);

    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;
    let comparison = analysis::analyze_adaptive_comparison(
        nmos,
        pmos,
        diff_point,
        mirror_point,
        TAIL_CURRENT / 2.0,
        &manifest.join("examples/ota_4t"),
        &output_root.join("mna"),
    )?;
    validate_finger_width("simplediffpair", comparison.diff_pair)?;
    validate_finger_width("currentmirror", comparison.current_mirror)?;
    validate_loop_metric_pairs(comparison.dense_metrics, comparison.adaptive_metrics)?;
    print_sizing(&comparison, diff_point, mirror_point);

    let include_loop_metrics =
        has_loop_metrics(comparison.dense_metrics) || has_loop_metrics(comparison.adaptive_metrics);
    let input = verification_input(diff_point, &comparison)?;
    let report = VerificationEngine::new(verification_config(
        manifest,
        &output_root,
        comparison.current_mirror,
        include_loop_metrics,
    ))?
    .verify_one(&input)?;
    let run = report
        .runs
        .first()
        .ok_or_else(|| io::Error::other("verification produced no run artifacts"))?;
    let simulation = report
        .rows
        .iter()
        .map(|row| (row.metric.as_str(), row.simulation_value))
        .collect::<BTreeMap<_, _>>();
    let rows = comparison_rows(
        comparison.adaptive_metrics,
        comparison.dense_metrics,
        &simulation,
    );

    let dense_ac_path = run.artifact_dir.join("shapeic_dense_ac.csv");
    let adaptive_ac_path = run.artifact_dir.join("shapeic_adaptive_ac.csv");
    let comparison_csv_path = run.artifact_dir.join("adaptive_comparison.csv");
    comparison.dense_sweep.write_csv(&dense_ac_path)?;
    comparison.adaptive_sweep.write_csv(&adaptive_ac_path)?;
    write_comparison_csv(&comparison_csv_path, &rows)?;
    let plot_helper = manifest.join("examples/ota_4t_electrical_verification/plot_ac.py");

    println!();
    println!("Shapeic adaptive vs. dense MNA vs. transistor-level ngspice");
    print!("{}", render_comparison_table(&rows));
    println!(
        "Adaptive frequency evaluations: {} (dense: {})",
        comparison.adaptive_frequency_evaluations,
        analysis::AC_POINTS_PER_DECADE * 11 + 1
    );
    println!("NGSpice CSV: {}", report.summary_path.display());
    println!("Three-way CSV: {}", comparison_csv_path.display());
    println!(
        "Rendered NGSpice netlist: {}",
        run.artifact_dir.join("netlist.spice").display()
    );
    println!(
        "NGSpice operating-point log: {}",
        run.artifact_dir.join("ngspice.log").display()
    );
    println!("Shapeic dense AC sweep: {}", dense_ac_path.display());
    println!("Shapeic adaptive AC sweep: {}", adaptive_ac_path.display());
    println!(
        "NGSpice AC sweep: {}",
        run.artifact_dir.join("ngspice_ac.tsv").display()
    );
    println!(
        "AC comparison plot: python3 {} {} --shapeic-file shapeic_dense_ac.csv \
         --adaptive-file shapeic_adaptive_ac.csv --title \"4T OTA adaptive AC verification\"",
        plot_helper.display(),
        run.artifact_dir.display()
    );
    Ok(())
}

fn verification_input(
    diff_point: OperatingPoint,
    comparison: &ElectricalAdaptiveComparison,
) -> Result<VerificationInput, io::Error> {
    let mut input = VerificationInput::new(
        diff_point,
        comparison.diff_pair.total_width,
        comparison.diff_pair.nf,
    );
    for metric in METRICS {
        if let Some(value) = metric_value(comparison.adaptive_metrics, metric)
            .or_else(|| metric_value(comparison.dense_metrics, metric))
        {
            input = input.reference(metric, value);
        }
    }
    if input.reference_values.is_empty() {
        return Err(io::Error::other(
            "adaptive verification produced no reference metrics",
        ));
    }
    Ok(input)
}

#[derive(Clone, Debug)]
struct ComparisonRow {
    metric: &'static str,
    adaptive: Option<f64>,
    dense: Option<f64>,
    ngspice: Option<f64>,
    adaptive_dense: ErrorValues,
    adaptive_ngspice: ErrorValues,
    dense_ngspice: ErrorValues,
}

#[derive(Clone, Copy, Debug, Default)]
struct ErrorValues {
    absolute: Option<f64>,
    percent: Option<f64>,
}

fn comparison_rows(
    adaptive: AcMetrics,
    dense: AcMetrics,
    simulation: &BTreeMap<&str, Option<f64>>,
) -> Vec<ComparisonRow> {
    METRICS
        .into_iter()
        .map(|metric| {
            let adaptive = metric_value(adaptive, metric);
            let dense = metric_value(dense, metric);
            let ngspice = simulation.get(metric).copied().flatten();
            ComparisonRow {
                metric,
                adaptive,
                dense,
                ngspice,
                adaptive_dense: error_values(adaptive, dense),
                adaptive_ngspice: error_values(adaptive, ngspice),
                dense_ngspice: error_values(dense, ngspice),
            }
        })
        .collect()
}

fn metric_value(metrics: AcMetrics, metric: &str) -> Option<f64> {
    match metric {
        "dc_gain_db" => Some(metrics.dc_gain_db),
        "bandwidth_3db_hz" => metrics.bandwidth_3db_hz,
        "unity_gain_hz" => metrics.unity_gain_hz,
        "phase_margin_deg" => metrics.phase_margin_deg,
        _ => None,
    }
}

fn error_values(compared: Option<f64>, reference: Option<f64>) -> ErrorValues {
    let (Some(compared), Some(reference)) = (compared, reference) else {
        return ErrorValues::default();
    };
    let absolute = (compared - reference).abs();
    ErrorValues {
        absolute: Some(absolute),
        percent: (reference != 0.0).then_some(100.0 * absolute / reference.abs()),
    }
}

fn render_comparison_table(rows: &[ComparisonRow]) -> String {
    const HEADERS: [&str; 10] = [
        "metric", "adaptive", "dense", "ngspice", "|A-D|", "A-D [%]", "|A-N|", "A-N [%]", "|D-N|",
        "D-N [%]",
    ];
    let rows = rows
        .iter()
        .map(|row| {
            [
                row.metric.to_owned(),
                format_optional(row.adaptive),
                format_optional(row.dense),
                format_optional(row.ngspice),
                format_optional(row.adaptive_dense.absolute),
                format_optional(row.adaptive_dense.percent),
                format_optional(row.adaptive_ngspice.absolute),
                format_optional(row.adaptive_ngspice.percent),
                format_optional(row.dense_ngspice.absolute),
                format_optional(row.dense_ngspice.percent),
            ]
        })
        .collect::<Vec<_>>();
    let widths = std::array::from_fn::<_, 10, _>(|column| {
        rows.iter()
            .map(|row| row[column].len())
            .max()
            .unwrap_or(0)
            .max(HEADERS[column].len())
    });
    let mut output = String::new();
    write_border(&mut output, &widths);
    write_row(&mut output, &HEADERS, &widths);
    write_border(&mut output, &widths);
    for row in &rows {
        let values = row.each_ref().map(String::as_str);
        write_row(&mut output, &values, &widths);
    }
    write_border(&mut output, &widths);
    output
}

fn write_border(output: &mut String, widths: &[usize; 10]) {
    output.push('+');
    for width in widths {
        output.extend(std::iter::repeat_n('-', width + 2));
        output.push('+');
    }
    output.push('\n');
}

fn write_row(output: &mut String, values: &[&str; 10], widths: &[usize; 10]) {
    output.push('|');
    for (value, width) in values.iter().zip(widths) {
        write!(output, " {value:<width$} |").expect("writing to a String cannot fail");
    }
    output.push('\n');
}

fn write_comparison_csv(path: &Path, rows: &[ComparisonRow]) -> Result<(), io::Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "metric,adaptive,dense,ngspice,adaptive_dense_absolute,\
         adaptive_dense_percent,adaptive_ngspice_absolute,adaptive_ngspice_percent,\
         dense_ngspice_absolute,dense_ngspice_percent"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            row.metric,
            csv_optional(row.adaptive),
            csv_optional(row.dense),
            csv_optional(row.ngspice),
            csv_optional(row.adaptive_dense.absolute),
            csv_optional(row.adaptive_dense.percent),
            csv_optional(row.adaptive_ngspice.absolute),
            csv_optional(row.adaptive_ngspice.percent),
            csv_optional(row.dense_ngspice.absolute),
            csv_optional(row.dense_ngspice.percent),
        )?;
    }
    writer.flush()
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.6e}"))
}

fn csv_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.17e}"))
}

fn validate_loop_metric_pairs(dense: AcMetrics, adaptive: AcMetrics) -> Result<(), io::Error> {
    for (name, metrics) in [("dense", dense), ("adaptive", adaptive)] {
        if metrics.unity_gain_hz.is_some() != metrics.phase_margin_deg.is_some() {
            return Err(io::Error::other(format!(
                "{name} analysis returned only one of UGF and phase margin"
            )));
        }
    }
    Ok(())
}

fn has_loop_metrics(metrics: AcMetrics) -> bool {
    metrics.unity_gain_hz.is_some() && metrics.phase_margin_deg.is_some()
}

fn print_sizing(
    comparison: &ElectricalAdaptiveComparison,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
) {
    println!("Adaptive electrical OTA verification point");
    println!(
        "{:<16} | {:>9} | {:>9} | {:>9} | {:>9} | {:>5} | {:>11} | {:>10} | {:>10} | {:>10}",
        "primitive",
        "L [um]",
        "VGS [V]",
        "VDS [V]",
        "Wf [um]",
        "nf",
        "W total [um]",
        "Ireq [uA]",
        "Id [uA]",
        "error [%]",
    );
    println!(
        "-----------------+-----------+-----------+-----------+-----------+-------+-------------+------------+------------+-----------"
    );
    print_sizing_row("simplediffpair", comparison.diff_pair, diff_point);
    print_sizing_row("currentmirror", comparison.current_mirror, mirror_point);
}

fn print_sizing_row(name: &str, sizing: SizingSummary, point: OperatingPoint) {
    println!(
        "{name:<16} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>5} | {:>11.3} | {:>10.3} | {:>10.3} | {:>+10.3e}",
        sizing.length * 1.0e6,
        point.vgs,
        point.vds,
        sizing.finger_width * 1.0e6,
        sizing.nf,
        sizing.total_width * 1.0e6,
        sizing.requested_current * 1.0e6,
        sizing.predicted_current * 1.0e6,
        sizing.relative_error_percent()
    );
}

fn lut_paths() -> Result<(PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_electrical_adaptive_verification"));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_point_has_all_expected_biases() {
        let diff = OperatingPoint::new(
            DIFF_LENGTH,
            0.0,
            VG_DC - SOURCE_VOLTAGE,
            VOUT_DC - SOURCE_VOLTAGE,
        );
        assert!((diff.vgs - 0.227_177_177_177_177_2).abs() < 1.0e-12);
        assert!((diff.vds - 0.327_177_177_177_177_2).abs() < 1.0e-12);
    }

    #[test]
    fn comparison_errors_use_dense_or_ngspice_as_the_reference() {
        let error = error_values(Some(11.0), Some(10.0));
        assert_eq!(error.absolute, Some(1.0));
        assert_eq!(error.percent, Some(10.0));
        assert_eq!(error_values(None, Some(10.0)).absolute, None);
    }
}
