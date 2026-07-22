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
