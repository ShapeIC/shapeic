use std::fmt::Write;

use super::VerificationReport;

const HEADERS: [&str; 13] = [
    "point",
    "status",
    "L [um]",
    "VBS [V]",
    "VGS [V]",
    "VDS [V]",
    "W [um]",
    "nf",
    "metric",
    "LUT/manual",
    "ngspice",
    "abs error",
    "error [%]",
];

impl VerificationReport {
    /// Render comparison rows as an aligned ASCII table suitable for a terminal.
    pub fn render_table(&self) -> String {
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let point = row.operating_point;
                [
                    row.point_index.to_string(),
                    row.status.as_str().to_owned(),
                    format_decimal(point.length * 1.0e6),
                    format_decimal(point.vbs),
                    format_decimal(point.vgs),
                    format_decimal(point.vds),
                    format_decimal(row.width * 1.0e6),
                    row.nf.to_string(),
                    row.metric.clone(),
                    format_scientific(row.reference_value),
                    optional_scientific(row.simulation_value),
                    optional_scientific(row.absolute_error),
                    optional_scientific(row.percent_error),
                ]
            })
            .collect::<Vec<_>>();

        render_ascii_table(&rows)
    }
}

fn format_decimal(value: f64) -> String {
    format!("{value:.6}")
}

fn format_scientific(value: f64) -> String {
    format!("{value:.6e}")
}

fn optional_scientific(value: Option<f64>) -> String {
    value
        .map(format_scientific)
        .unwrap_or_else(|| "-".to_owned())
}

fn render_ascii_table(rows: &[[String; HEADERS.len()]]) -> String {
    let widths = std::array::from_fn::<_, { HEADERS.len() }, _>(|column| {
        rows.iter()
            .map(|row| row[column].len())
            .max()
            .unwrap_or(0)
            .max(HEADERS[column].len())
    });
    let border = table_border(&widths);
    let mut output = String::new();

    output.push_str(&border);
    write_table_row(&mut output, &HEADERS, &widths);
    output.push_str(&border);
    for row in rows {
        let values = row.each_ref().map(String::as_str);
        write_table_row(&mut output, &values, &widths);
    }
    output.push_str(&border);

    output
}

fn table_border(widths: &[usize; HEADERS.len()]) -> String {
    let mut border = String::new();
    border.push('+');
    for width in widths {
        border.extend(std::iter::repeat_n('-', width + 2));
        border.push('+');
    }
    border.push('\n');
    border
}

fn write_table_row(
    output: &mut String,
    values: &[&str; HEADERS.len()],
    widths: &[usize; HEADERS.len()],
) {
    output.push('|');
    for (value, width) in values.iter().zip(widths) {
        write!(output, " {value:<width$} |").expect("writing to a String cannot fail");
    }
    output.push('\n');
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::OperatingPoint;
    use crate::verification::{MetricStatus, VerificationRow};

    #[test]
    fn renders_aligned_comparison_rows_and_missing_values() {
        let point = OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6);
        let report = VerificationReport {
            output_dir: PathBuf::from("output"),
            summary_path: PathBuf::from("output/summary.csv"),
            runs: Vec::new(),
            rows: vec![
                VerificationRow {
                    point_index: 0,
                    operating_point: point,
                    width: 5.0e-6,
                    nf: 1,
                    metric: "id".to_owned(),
                    status: MetricStatus::Compared,
                    reference_value: 1.0e-3,
                    simulation_value: Some(1.1e-3),
                    absolute_error: Some(1.0e-4),
                    percent_error: Some(10.0),
                    message: None,
                },
                VerificationRow {
                    point_index: 0,
                    operating_point: point,
                    width: 5.0e-6,
                    nf: 1,
                    metric: "vth".to_owned(),
                    status: MetricStatus::SimulationError,
                    reference_value: 0.3,
                    simulation_value: None,
                    absolute_error: None,
                    percent_error: None,
                    message: Some("missing column".to_owned()),
                },
            ],
        };

        let table = report.render_table();
        assert!(table.contains("| point | status"));
        assert!(table.contains("| LUT/manual"));
        assert!(table.contains("1.000000e-3"));
        assert!(table.contains("1.100000e-3"));
        assert!(table.contains("simulation_error"));
        assert!(table.contains("| -"));

        let line_widths = table.lines().map(str::len).collect::<Vec<_>>();
        assert!(line_widths.windows(2).all(|pair| pair[0] == pair[1]));
    }
}
