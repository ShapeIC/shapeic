use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{
    MetricStatus, VerificationConfig, VerificationError, VerificationInput, VerificationReport,
    VerificationRow, VerificationRun, VerificationStatus, template,
};

const DYNAMIC_TEMPLATE_KEYS: [&str; 7] =
    ["width", "nf", "length", "vbs", "vgs", "vds", "results_path"];

/// Sequential NGSpice runner for checking manually calculated operating points.
#[derive(Clone, Debug)]
pub struct VerificationEngine {
    config: VerificationConfig,
    template_source: String,
}

impl VerificationEngine {
    pub fn new(config: VerificationConfig) -> Result<Self, VerificationError> {
        validate_config(&config)?;
        let template_source = fs::read_to_string(&config.template).map_err(|source| {
            VerificationError::io(
                format!(
                    "failed to read verification template '{}'",
                    config.template.display()
                ),
                source,
            )
        })?;

        let mut validation_values = config.template_variables.clone();
        for key in DYNAMIC_TEMPLATE_KEYS {
            validation_values.insert(key.to_owned(), "0".to_owned());
        }
        template::render(&template_source, &validation_values)?;

        Ok(Self {
            config,
            template_source,
        })
    }

    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }

    pub fn verify_one(
        &self,
        input: &VerificationInput,
    ) -> Result<VerificationReport, VerificationError> {
        self.verify(std::slice::from_ref(input))
    }

    pub fn verify(
        &self,
        inputs: &[VerificationInput],
    ) -> Result<VerificationReport, VerificationError> {
        validate_inputs(&self.config, inputs)?;
        let output_dir = create_output_dir(&self.config.output_dir)?;
        let summary_path = output_dir.join("summary.csv");
        let mut csv = csv::Writer::from_path(&summary_path)?;
        csv.write_record([
            "point_index",
            "run_status",
            "metric_status",
            "length",
            "vbs",
            "vgs",
            "vds",
            "width",
            "nf",
            "metric",
            "reference_value",
            "simulation_value",
            "absolute_error",
            "percent_error",
            "message",
        ])?;

        let row_count = inputs
            .iter()
            .map(|input| input.reference_values.len())
            .sum();
        let mut runs = Vec::with_capacity(inputs.len());
        let mut rows = Vec::with_capacity(row_count);
        for (point_index, input) in inputs.iter().enumerate() {
            let (run, point_rows) = self.verify_input(point_index, input, &output_dir);
            for row in &point_rows {
                write_csv_row(&mut csv, run.status, row)?;
            }
            csv.flush().map_err(|source| {
                VerificationError::io(
                    format!("failed to flush '{}'", summary_path.display()),
                    source,
                )
            })?;
            runs.push(run);
            rows.extend(point_rows);
        }

        Ok(VerificationReport {
            output_dir,
            summary_path,
            runs,
            rows,
        })
    }

    fn verify_input(
        &self,
        point_index: usize,
        input: &VerificationInput,
        output_dir: &Path,
    ) -> (VerificationRun, Vec<VerificationRow>) {
        let artifact_dir = output_dir.join(format!("point_{point_index:06}"));
        if let Err(source) = fs::create_dir(&artifact_dir) {
            return failed_run(
                point_index,
                input,
                artifact_dir,
                format!("failed to create point artifact directory: {source}"),
            );
        }

        let results_path = artifact_dir.join("results.tsv");
        let netlist_path = artifact_dir.join("netlist.spice");
        let log_path = artifact_dir.join("ngspice.log");
        let context = render_context(&self.config, input, Path::new("results.tsv"));
        let netlist = match template::render(&self.template_source, &context) {
            Ok(netlist) => netlist,
            Err(error) => {
                return failed_run(point_index, input, artifact_dir, error.to_string());
            }
        };
        if let Err(source) = fs::write(&netlist_path, netlist) {
            return failed_run(
                point_index,
                input,
                artifact_dir,
                format!("failed to write netlist: {source}"),
            );
        }

        let process = Command::new(&self.config.ngspice)
            .arg("-n")
            .arg("-b")
            .arg("-o")
            .arg(&log_path)
            .arg(&netlist_path)
            .envs(&self.config.environment)
            .current_dir(&artifact_dir)
            .output();
        let output = match process {
            Ok(output) => output,
            Err(source) => {
                let message = format!(
                    "failed to launch '{}': {source}",
                    self.config.ngspice.display()
                );
                let _ = fs::write(&log_path, &message);
                return failed_run(point_index, input, artifact_dir, message);
            }
        };
        if !output.status.success() {
            return failed_run(
                point_index,
                input,
                artifact_dir,
                format!("NGSpice exited with status {}", output.status),
            );
        }

        let columns = match parse_wrdata(&results_path) {
            Ok(columns) => columns,
            Err(message) => return failed_run(point_index, input, artifact_dir, message),
        };
        let mut compared = 0_usize;
        let rows = input
            .reference_values
            .iter()
            .map(|(metric, reference_value)| {
                let row = comparison_row(
                    point_index,
                    input,
                    metric,
                    *reference_value,
                    columns.get(metric).copied(),
                );
                if row.status == MetricStatus::Compared {
                    compared += 1;
                }
                row
            })
            .collect::<Vec<_>>();
        let status = if compared == input.reference_values.len() {
            VerificationStatus::Success
        } else if compared == 0 {
            VerificationStatus::Failed
        } else {
            VerificationStatus::Partial
        };
        let message = (status != VerificationStatus::Success).then(|| {
            format!(
                "compared {compared} of {} reference values",
                input.reference_values.len()
            )
        });
        (
            VerificationRun {
                point_index,
                input: input.clone(),
                status,
                artifact_dir,
                message,
            },
            rows,
        )
    }
}

fn validate_config(config: &VerificationConfig) -> Result<(), VerificationError> {
    if config.ngspice.as_os_str().is_empty() {
        return Err(VerificationError::InvalidConfig(
            "NGSpice executable must not be empty".to_owned(),
        ));
    }
    if config.template.as_os_str().is_empty() {
        return Err(VerificationError::InvalidConfig(
            "template path must not be empty".to_owned(),
        ));
    }
    if config.output_dir.as_os_str().is_empty() {
        return Err(VerificationError::InvalidConfig(
            "output directory must not be empty".to_owned(),
        ));
    }
    if let Some(maximum) = config.max_width_per_finger
        && (!maximum.is_finite() || maximum <= 0.0)
    {
        return Err(VerificationError::InvalidConfig(
            "maximum width per finger must be finite and greater than zero".to_owned(),
        ));
    }
    for key in DYNAMIC_TEMPLATE_KEYS {
        if config.template_variables.contains_key(key) {
            return Err(VerificationError::InvalidConfig(format!(
                "template variable '{key}' is reserved"
            )));
        }
    }
    Ok(())
}

fn validate_inputs(
    config: &VerificationConfig,
    inputs: &[VerificationInput],
) -> Result<(), VerificationError> {
    if inputs.is_empty() {
        return Err(VerificationError::InvalidConfig(
            "at least one verification input is required".to_owned(),
        ));
    }
    for (index, input) in inputs.iter().enumerate() {
        let point = input.operating_point;
        if !input.width.is_finite() || input.width <= 0.0 {
            return Err(VerificationError::InvalidConfig(format!(
                "input {index} width must be finite and greater than zero"
            )));
        }
        if input.nf == 0 {
            return Err(VerificationError::InvalidConfig(format!(
                "input {index} nf must be greater than zero"
            )));
        }
        if let Some(maximum) = config.max_width_per_finger {
            let width_per_finger = input.width / f64::from(input.nf);
            if width_per_finger >= maximum {
                return Err(VerificationError::InvalidConfig(format!(
                    "input {index} width per finger ({width_per_finger:.17e}) must be less than {maximum:.17e}"
                )));
            }
        }
        if !point.length.is_finite() || point.length <= 0.0 {
            return Err(VerificationError::InvalidConfig(format!(
                "input {index} length must be finite and greater than zero"
            )));
        }
        for (name, value) in [("vbs", point.vbs), ("vgs", point.vgs), ("vds", point.vds)] {
            if !value.is_finite() {
                return Err(VerificationError::InvalidConfig(format!(
                    "input {index} {name} must be finite"
                )));
            }
        }
        if input.reference_values.is_empty() {
            return Err(VerificationError::InvalidConfig(format!(
                "input {index} must contain at least one reference value"
            )));
        }
        for (name, value) in &input.reference_values {
            if name.is_empty() || name.chars().any(char::is_whitespace) {
                return Err(VerificationError::InvalidConfig(format!(
                    "input {index} reference names must be non-empty and contain no whitespace"
                )));
            }
            if !value.is_finite() {
                return Err(VerificationError::InvalidConfig(format!(
                    "input {index} reference '{name}' must be finite"
                )));
            }
        }
    }
    Ok(())
}

fn create_output_dir(path: &Path) -> Result<PathBuf, VerificationError> {
    if path.exists() {
        return Err(VerificationError::OutputExists(path.to_owned()));
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| {
            VerificationError::io(
                format!("failed to create output parent '{}'", parent.display()),
                source,
            )
        })?;
    }
    fs::create_dir(path).map_err(|source| {
        VerificationError::io(
            format!("failed to create output directory '{}'", path.display()),
            source,
        )
    })?;
    path.canonicalize().map_err(|source| {
        VerificationError::io(
            format!("failed to resolve output directory '{}'", path.display()),
            source,
        )
    })
}

fn render_context(
    config: &VerificationConfig,
    input: &VerificationInput,
    results_path: &Path,
) -> BTreeMap<String, String> {
    let point = input.operating_point;
    let mut context = config.template_variables.clone();
    context.insert("width".to_owned(), format_float(input.width));
    context.insert("nf".to_owned(), input.nf.to_string());
    context.insert("length".to_owned(), format_float(point.length));
    context.insert("vbs".to_owned(), format_float(point.vbs));
    context.insert("vgs".to_owned(), format_float(point.vgs));
    context.insert("vds".to_owned(), format_float(point.vds));
    context.insert(
        "results_path".to_owned(),
        results_path.display().to_string(),
    );
    context
}

fn format_float(value: f64) -> String {
    format!("{value:.17e}")
}

fn parse_wrdata(path: &Path) -> Result<BTreeMap<String, f64>, String> {
    let text = fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read NGSpice results '{}': {error}",
            path.display()
        )
    })?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let headers = lines
        .next()
        .ok_or_else(|| "NGSpice results contain no header".to_owned())?
        .split_whitespace()
        .collect::<Vec<_>>();
    let values = lines
        .next()
        .ok_or_else(|| "NGSpice results contain no data row".to_owned())?
        .split_whitespace()
        .collect::<Vec<_>>();
    if lines.next().is_some() {
        return Err("NGSpice results must contain exactly one data row".to_owned());
    }
    if headers.len() != values.len() {
        return Err(format!(
            "NGSpice results have {} headers but {} values",
            headers.len(),
            values.len()
        ));
    }
    headers
        .into_iter()
        .zip(values)
        .map(|(header, value)| {
            let value = value.parse::<f64>().map_err(|error| {
                format!("invalid NGSpice value '{value}' for column '{header}': {error}")
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "non-finite NGSpice value '{value}' for column '{header}'"
                ));
            }
            Ok((header.to_owned(), value))
        })
        .collect()
}

fn comparison_row(
    point_index: usize,
    input: &VerificationInput,
    metric: &str,
    reference_value: f64,
    simulation_value: Option<f64>,
) -> VerificationRow {
    match simulation_value {
        Some(simulation_value) => {
            let absolute_error = (reference_value - simulation_value).abs();
            let percent_error = (simulation_value != 0.0)
                .then_some(100.0 * absolute_error / simulation_value.abs());
            VerificationRow {
                point_index,
                operating_point: input.operating_point,
                width: input.width,
                nf: input.nf,
                metric: metric.to_owned(),
                status: MetricStatus::Compared,
                reference_value,
                simulation_value: Some(simulation_value),
                absolute_error: Some(absolute_error),
                percent_error,
                message: None,
            }
        }
        None => VerificationRow {
            point_index,
            operating_point: input.operating_point,
            width: input.width,
            nf: input.nf,
            metric: metric.to_owned(),
            status: MetricStatus::SimulationError,
            reference_value,
            simulation_value: None,
            absolute_error: None,
            percent_error: None,
            message: Some(format!("NGSpice results do not contain column '{metric}'")),
        },
    }
}

fn failed_run(
    point_index: usize,
    input: &VerificationInput,
    artifact_dir: PathBuf,
    message: String,
) -> (VerificationRun, Vec<VerificationRow>) {
    let rows = input
        .reference_values
        .iter()
        .map(|(metric, reference_value)| VerificationRow {
            point_index,
            operating_point: input.operating_point,
            width: input.width,
            nf: input.nf,
            metric: metric.clone(),
            status: MetricStatus::SimulationError,
            reference_value: *reference_value,
            simulation_value: None,
            absolute_error: None,
            percent_error: None,
            message: Some(message.clone()),
        })
        .collect();
    (
        VerificationRun {
            point_index,
            input: input.clone(),
            status: VerificationStatus::Failed,
            artifact_dir,
            message: Some(message),
        },
        rows,
    )
}

fn write_csv_row(
    writer: &mut csv::Writer<fs::File>,
    run_status: VerificationStatus,
    row: &VerificationRow,
) -> Result<(), csv::Error> {
    let point = row.operating_point;
    writer.write_record([
        row.point_index.to_string(),
        run_status.as_str().to_owned(),
        row.status.as_str().to_owned(),
        format_float(point.length),
        format_float(point.vbs),
        format_float(point.vgs),
        format_float(point.vds),
        format_float(row.width),
        row.nf.to_string(),
        row.metric.clone(),
        format_float(row.reference_value),
        optional_float(row.simulation_value),
        optional_float(row.absolute_error),
        optional_float(row.percent_error),
        row.message.clone().unwrap_or_default(),
    ])
}

fn optional_float(value: Option<f64>) -> String {
    value.map(format_float).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OperatingPoint;

    #[test]
    fn parses_single_wrdata_row_and_rejects_multiple_rows() {
        let root = std::env::temp_dir().join(format!(
            "shapeic-lut-wrdata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir(&root).expect("temp directory");
        let path = root.join("results.tsv");
        fs::write(
            &path,
            "scale id gm cgg cgd\n0 1e-3 2e-3 4.5e-15 -2.25e-16\n",
        )
        .expect("fixture");
        let values = parse_wrdata(&path).expect("parse");
        assert_eq!(values["id"], 1.0e-3);
        assert_eq!(values["cgg"], 4.5e-15);
        assert_eq!(values["cgd"], -2.25e-16);

        fs::write(&path, "scale id\n0 1\n1 2\n").expect("fixture");
        assert!(parse_wrdata(&path).is_err());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn percent_error_is_empty_for_zero_simulation_reference() {
        let input = VerificationInput::new(OperatingPoint::new(1.0, 0.0, 1.0, 1.0), 1.0, 1)
            .reference("value", 1.0);
        let row = comparison_row(0, &input, "value", 1.0, Some(0.0));
        assert_eq!(row.absolute_error, Some(1.0));
        assert_eq!(row.percent_error, None);
    }

    #[test]
    fn width_per_finger_limit_is_optional_and_strict() {
        let point = OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0);
        let at_limit = VerificationInput::new(point, 20.0e-6, 2).reference("id", 1.0);
        let below_limit = VerificationInput::new(point, 19.0e-6, 2).reference("id", 1.0);
        let unlimited = VerificationConfig::new("template", "output");
        assert!(validate_inputs(&unlimited, std::slice::from_ref(&at_limit)).is_ok());

        let limited = unlimited.max_width_per_finger(10.0e-6);
        assert!(validate_inputs(&limited, &[at_limit]).is_err());
        assert!(validate_inputs(&limited, &[below_limit]).is_ok());
    }
}
