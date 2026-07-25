use std::collections::BTreeMap;
use std::fmt;
use std::io::{Read, Seek};
use std::path::Path;

use crate::{Expr, LutArray, LutError, MosExpression};

/// One of the four independent axes in an SSTADEx MOS lookup table.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Axis {
    Length,
    Vbs,
    Vgs,
    Vds,
}

impl Axis {
    pub const ALL: [Self; 4] = [Self::Length, Self::Vbs, Self::Vgs, Self::Vds];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Vbs => "vbs",
            Self::Vgs => "vgs",
            Self::Vds => "vds",
        }
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Length => 0,
            Self::Vbs => 1,
            Self::Vgs => 2,
            Self::Vds => 3,
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Physical coordinates used to query a device LUT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperatingPoint {
    pub length: f64,
    pub vbs: f64,
    pub vgs: f64,
    pub vds: f64,
}

impl OperatingPoint {
    pub const fn new(length: f64, vbs: f64, vgs: f64, vds: f64) -> Self {
        Self {
            length,
            vbs,
            vgs,
            vds,
        }
    }

    pub const fn value(self, axis: Axis) -> f64 {
        match axis {
            Axis::Length => self.length,
            Axis::Vbs => self.vbs,
            Axis::Vgs => self.vgs,
            Axis::Vds => self.vds,
        }
    }
}

/// Five-dimensional coordinates used to query a width-dependent LUT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LutPoint {
    pub operating_point: OperatingPoint,
    pub finger_width: f64,
}

/// Intrinsic MOS nodal capacitance matrix in terminal order G, D, S, B.
///
/// The values are ready to stamp as the coefficient of `s` in MNA. Unlike an
/// extracted passive interconnect matrix, an intrinsic MOS matrix need not be
/// symmetric, but every row and column must conserve charge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosCapacitanceMatrix {
    pub values: [[f64; 4]; 4],
}

impl MosCapacitanceMatrix {
    pub const TERMINALS: [&'static str; 4] = ["G", "D", "S", "B"];
    pub const PARAMETERS: [&'static str; 16] = [
        "cgg", "cgd", "cgs", "cgb", "cdg", "cdd", "cds", "cdb", "csg", "csd", "css", "csb", "cbg",
        "cbd", "cbs", "cbb",
    ];

    /// Convert the sixteen raw ngspice coefficients into a nodal matrix.
    ///
    /// Ngspice reports mutual coefficients using the compact-model convention;
    /// their signs must be inverted before they are stamped in MNA.
    pub fn from_ngspice_parameters(values: &[f64]) -> Result<Self, LutError> {
        if values.len() != 16 {
            return Err(LutError::InvalidCapacitanceMatrix {
                reason: format!("expected 16 values, found {}", values.len()),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(LutError::InvalidCapacitanceMatrix {
                reason: "all coefficients must be finite".to_owned(),
            });
        }
        let matrix = Self {
            values: std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let value = values[row * 4 + column];
                    if row == column { value } else { -value }
                })
            }),
        };
        matrix.validate_charge_conservation()?;
        Ok(matrix)
    }

    /// Alias for [`Self::from_ngspice_parameters`].
    ///
    /// The flat values must follow [`Self::PARAMETERS`] and use the raw
    /// ngspice convention, not an already signed nodal-matrix convention.
    pub fn from_flat(values: &[f64]) -> Result<Self, LutError> {
        Self::from_ngspice_parameters(values)
    }

    pub fn scaled(self, factor: f64) -> Self {
        Self {
            values: self
                .values
                .map(|row| row.map(|coefficient| coefficient * factor)),
        }
    }

    fn validate_charge_conservation(&self) -> Result<(), LutError> {
        let scale = self
            .values
            .iter()
            .flatten()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
            .max(1.0e-30);
        let tolerance = scale * 1.0e-5;
        for index in 0..4 {
            let row_sum = self.values[index].iter().sum::<f64>();
            let column_sum = self.values.iter().map(|row| row[index]).sum::<f64>();
            if row_sum.abs() > tolerance || column_sum.abs() > tolerance {
                return Err(LutError::InvalidCapacitanceMatrix {
                    reason: format!(
                        "charge conservation failed at index {index}: row sum={row_sum:.6e}, column sum={column_sum:.6e}, tolerance={tolerance:.6e}"
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Total extrinsic MOS capacitances for a complete multi-finger device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosExtrinsicCapacitances {
    pub cgsol: f64,
    pub cgdol: f64,
    pub cjs: f64,
    pub cjd: f64,
}

impl MosExtrinsicCapacitances {
    pub const PARAMETERS: [&'static str; 4] = ["cgsol", "cgdol", "cjs", "cjd"];
    pub const NF_SAMPLES: [u32; 4] = [1, 2, 3, 4];
    pub const SAMPLE_PARAMETERS: [[&'static str; 4]; 4] = [
        ["cgsol", "cgsol_nf2", "cgsol_nf3", "cgsol_nf4"],
        ["cgdol", "cgdol_nf2", "cgdol_nf3", "cgdol_nf4"],
        ["cjs", "cjs_nf2", "cjs_nf3", "cjs_nf4"],
        ["cjd", "cjd_nf2", "cjd_nf3", "cjd_nf4"],
    ];

    pub(crate) fn from_nf_samples(nf: u32, values: &[f64]) -> Result<Self, LutError> {
        if nf == 0 {
            return Err(LutError::InvalidExtrinsicCapacitances {
                reason: "nf must be greater than zero".to_owned(),
            });
        }
        if values.len() != 16 {
            return Err(LutError::InvalidExtrinsicCapacitances {
                reason: format!("expected 16 sampled values, found {}", values.len()),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(LutError::InvalidExtrinsicCapacitances {
                reason: "all sampled values must be finite".to_owned(),
            });
        }

        let evaluated: [f64; 4] = std::array::from_fn(|parameter| {
            evaluate_nf_branch(nf, &values[parameter * 4..parameter * 4 + 4])
        });
        if evaluated.iter().any(|value| !value.is_finite()) {
            return Err(LutError::InvalidExtrinsicCapacitances {
                reason: "finger-count extrapolation produced a non-finite value".to_owned(),
            });
        }
        Ok(Self {
            cgsol: evaluated[0],
            cgdol: evaluated[1],
            cjs: evaluated[2],
            cjd: evaluated[3],
        })
    }
}

fn evaluate_nf_branch(nf: u32, samples: &[f64]) -> f64 {
    if nf % 2 == 1 {
        samples[0] + f64::from((nf - 1) / 2) * (samples[2] - samples[0])
    } else {
        samples[1] + f64::from((nf - 2) / 2) * (samples[3] - samples[1])
    }
}

impl LutPoint {
    pub const fn new(operating_point: OperatingPoint, finger_width: f64) -> Self {
        Self {
            operating_point,
            finger_width,
        }
    }
}

/// File-level metadata stored by a LUT generator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LutMetadata {
    pub description: Option<String>,
    pub simulator: Option<String>,
}

/// A complete lookup table, potentially containing multiple device models.
#[derive(Clone, Debug)]
pub struct LookupTable {
    pub(crate) metadata: LutMetadata,
    pub(crate) models: BTreeMap<String, DeviceLut>,
}

impl LookupTable {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LutError> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(file)
    }

    pub fn from_reader<R>(reader: R) -> Result<Self, LutError>
    where
        R: Read + Seek,
    {
        crate::npz::read_lookup_table(reader)
    }

    pub fn metadata(&self) -> &LutMetadata {
        &self.metadata
    }

    pub fn description(&self) -> Option<&str> {
        self.metadata.description.as_deref()
    }

    pub fn simulator(&self) -> Option<&str> {
        self.metadata.simulator.as_deref()
    }

    pub fn model_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    pub fn models(&self) -> impl ExactSizeIterator<Item = (&str, &DeviceLut)> {
        self.models
            .iter()
            .map(|(name, model)| (name.as_str(), model))
    }

    pub fn model(&self, name: &str) -> Result<&DeviceLut, LutError> {
        self.models
            .get(name)
            .ok_or_else(|| LutError::UnknownModel(name.to_owned()))
    }
}

/// Lookup data and metadata for one MOS model.
#[derive(Clone, Debug)]
pub struct DeviceLut {
    pub(crate) name: String,
    pub(crate) axes: [Vec<f64>; 4],
    pub(crate) finger_widths: Option<Vec<f64>>,
    pub(crate) parameters: BTreeMap<String, LutArray>,
    pub(crate) parameter_names: Vec<String>,
    pub(crate) device_parameters: BTreeMap<String, f64>,
}

impl DeviceLut {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axis(&self, axis: Axis) -> &[f64] {
        &self.axes[axis.index()]
    }

    /// Width-per-finger axis for a five-dimensional Shapeic LUT.
    pub fn finger_widths(&self) -> Option<&[f64]> {
        self.finger_widths.as_deref()
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    pub fn parameters(&self) -> impl ExactSizeIterator<Item = (&str, &LutArray)> {
        self.parameters
            .iter()
            .map(|(name, array)| (name.as_str(), array))
    }

    pub fn array(&self, name: &str) -> Result<&LutArray, LutError> {
        self.parameters
            .get(name)
            .ok_or_else(|| LutError::UnknownParameter {
                model: self.name.clone(),
                parameter: name.to_owned(),
            })
    }

    pub fn device_parameters(&self) -> &BTreeMap<String, f64> {
        &self.device_parameters
    }

    pub fn device_parameter(&self, name: &str) -> Result<f64, LutError> {
        self.device_parameters
            .get(name)
            .copied()
            .ok_or_else(|| LutError::UnknownDeviceParameter {
                model: self.name.clone(),
                parameter: name.to_owned(),
            })
    }

    pub fn parameter_expression(&self, name: &str) -> Result<Expr, LutError> {
        self.array(name)?;
        Ok(Expr::parameter(name))
    }

    pub fn standard_expression(&self, expression: MosExpression) -> Result<Expr, LutError> {
        crate::expression::standard_expression(self, expression)
    }

    pub(crate) fn extrinsic_capacitance_sample_expressions(
        &self,
    ) -> Result<Option<Vec<Expr>>, LutError> {
        let has_sampled_parameters = MosExtrinsicCapacitances::SAMPLE_PARAMETERS
            .iter()
            .flatten()
            .filter(|name| name.contains("_nf"))
            .any(|name| self.parameters.contains_key(*name));
        if !has_sampled_parameters {
            return Ok(None);
        }

        let missing = MosExtrinsicCapacitances::SAMPLE_PARAMETERS
            .iter()
            .flatten()
            .filter(|name| !self.parameters.contains_key(**name))
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(LutError::IncompleteExtrinsicCapacitanceSamples {
                model: self.name.clone(),
                missing,
            });
        }

        MosExtrinsicCapacitances::SAMPLE_PARAMETERS
            .iter()
            .flatten()
            .map(|name| self.parameter_expression(name))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub(crate) fn finger_width_at(&self, index: &[usize]) -> Result<f64, LutError> {
        if let Some(widths) = &self.finger_widths {
            let width_index = *index.get(4).ok_or_else(|| {
                LutError::schema(
                    format!("model '{}'.finger_width", self.name),
                    "a five-dimensional expression was evaluated without a width index",
                )
            })?;
            return widths.get(width_index).copied().ok_or_else(|| {
                LutError::schema(
                    format!("model '{}'.finger_width", self.name),
                    format!("index {width_index} is outside shape [{}]", widths.len()),
                )
            });
        }

        self.device_parameter("w")
    }
}
