//! Compare manually supplied reference values with transistor-level NGSpice operating points.

mod engine;
mod error;
mod table;
mod template;

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

use crate::OperatingPoint;

pub use engine::VerificationEngine;
pub use error::VerificationError;

/// Physical inputs and user-calculated values for one NGSpice comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct VerificationInput {
    pub operating_point: OperatingPoint,
    pub width: f64,
    pub nf: u32,
    pub reference_values: BTreeMap<String, f64>,
}

impl VerificationInput {
    pub fn new(operating_point: OperatingPoint, width: f64, nf: u32) -> Self {
        Self {
            operating_point,
            width,
            nf,
            reference_values: BTreeMap::new(),
        }
    }

    pub fn reference(mut self, name: impl Into<String>, value: f64) -> Self {
        self.reference_values.insert(name.into(), value);
        self
    }

    pub fn references(mut self, values: BTreeMap<String, f64>) -> Self {
        self.reference_values = values;
        self
    }
}

/// Files and template values used by a verification run.
#[derive(Clone, Debug)]
pub struct VerificationConfig {
    pub ngspice: PathBuf,
    pub template: PathBuf,
    pub output_dir: PathBuf,
    pub max_width_per_finger: Option<f64>,
    pub template_variables: BTreeMap<String, String>,
    pub environment: BTreeMap<OsString, OsString>,
}

impl VerificationConfig {
    pub fn new(template: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            ngspice: PathBuf::from("ngspice"),
            template: template.into(),
            output_dir: output_dir.into(),
            max_width_per_finger: None,
            template_variables: BTreeMap::new(),
            environment: BTreeMap::new(),
        }
    }

    pub fn template_variable(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.template_variables.insert(name.into(), value.into());
        self
    }

    pub fn max_width_per_finger(mut self, value: f64) -> Self {
        self.max_width_per_finger = Some(value);
        self
    }

    pub fn environment(mut self, name: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.insert(name.into(), value.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationStatus {
    Success,
    Partial,
    Failed,
}

impl VerificationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Partial => "partial",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricStatus {
    Compared,
    SimulationError,
}

impl MetricStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compared => "compared",
            Self::SimulationError => "simulation_error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerificationRun {
    pub point_index: usize,
    pub input: VerificationInput,
    pub status: VerificationStatus,
    pub artifact_dir: PathBuf,
    pub message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct VerificationRow {
    pub point_index: usize,
    pub operating_point: OperatingPoint,
    pub width: f64,
    pub nf: u32,
    pub metric: String,
    pub status: MetricStatus,
    pub reference_value: f64,
    pub simulation_value: Option<f64>,
    pub absolute_error: Option<f64>,
    pub percent_error: Option<f64>,
    pub message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct VerificationReport {
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub runs: Vec<VerificationRun>,
    pub rows: Vec<VerificationRow>,
}
