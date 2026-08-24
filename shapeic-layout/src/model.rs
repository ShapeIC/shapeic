use std::collections::BTreeMap;

use ndarray::{Array2, Array5, ArrayD};

use crate::LayoutError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalPoint {
    pub length: f64,
    pub finger_width: f64,
    pub nf: u32,
}

impl PhysicalPoint {
    pub const fn new(length: f64, finger_width: f64, nf: u32) -> Self {
        Self {
            length,
            finger_width,
            nf,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutAwarePoint {
    pub length: f64,
    pub finger_width: f64,
    pub nf: u32,
    pub vbs: f64,
    pub vgs: f64,
    pub vds: f64,
}

impl LayoutAwarePoint {
    pub const fn new(
        length: f64,
        finger_width: f64,
        nf: u32,
        vbs: f64,
        vgs: f64,
        vds: f64,
    ) -> Self {
        Self {
            length,
            finger_width,
            nf,
            vbs,
            vgs,
            vds,
        }
    }

    pub const fn physical_point(self) -> PhysicalPoint {
        PhysicalPoint::new(self.length, self.finger_width, self.nf)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortAdmittance {
    pub ports: Vec<String>,
    pub conductance: Array2<f64>,
    pub capacitance: Array2<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutAwareAdmittance {
    pub ports: Vec<String>,
    pub interconnect_conductance: Array2<f64>,
    pub interconnect_capacitance: Array2<f64>,
    pub device_capacitance_correction: Array2<f64>,
}

impl LayoutAwareAdmittance {
    pub fn total_capacitance(&self) -> Array2<f64> {
        &self.interconnect_capacitance + &self.device_capacitance_correction
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMetadata {
    pub format_version: u32,
    pub pdk: String,
    pub layout_policy: String,
    pub generator: Option<String>,
}

#[derive(Debug)]
pub struct PhysicalLookupTable {
    pub(crate) metadata: PhysicalMetadata,
    pub(crate) primitives: BTreeMap<String, PhysicalPrimitive>,
}

impl PhysicalLookupTable {
    pub fn metadata(&self) -> &PhysicalMetadata {
        &self.metadata
    }

    pub fn primitive(&self, name: &str) -> Result<&PhysicalPrimitive, LayoutError> {
        self.primitives
            .get(name)
            .ok_or_else(|| LayoutError::UnknownPrimitive(name.to_owned()))
    }

    pub fn primitive_names(&self) -> impl Iterator<Item = &str> {
        self.primitives.keys().map(String::as_str)
    }
}

#[derive(Debug)]
pub struct PhysicalPrimitive {
    pub(crate) name: String,
    pub(crate) ports: Vec<String>,
    pub(crate) lengths: Vec<f64>,
    pub(crate) finger_widths: Vec<f64>,
    pub(crate) finger_counts: Vec<f64>,
    pub(crate) conductance: Array5<f64>,
    pub(crate) capacitance: Array5<f64>,
    pub(crate) device_capacitance_correction: Option<DeviceCapacitanceCorrection>,
}

#[derive(Debug)]
pub(crate) struct DeviceCapacitanceCorrection {
    pub(crate) finger_counts: Vec<f64>,
    pub(crate) vbs: Vec<f64>,
    pub(crate) vgs: Vec<f64>,
    pub(crate) vds: Vec<f64>,
    pub(crate) capacitance: ArrayD<f64>,
}

impl PhysicalPrimitive {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ports(&self) -> &[String] {
        &self.ports
    }

    pub fn lengths(&self) -> &[f64] {
        &self.lengths
    }

    pub fn finger_widths(&self) -> &[f64] {
        &self.finger_widths
    }

    pub fn finger_counts(&self) -> impl Iterator<Item = u32> + '_ {
        self.finger_counts.iter().map(|value| *value as u32)
    }
}
