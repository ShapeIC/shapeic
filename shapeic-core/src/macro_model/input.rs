use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use shapeic_lut::DeviceLut;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::exploration::candidate::CandidateSet;
use crate::exploration::filter::CandidateFilter;
use crate::primitive::build::PrimitiveBuildInput;

use super::Macro;

/// Build data and local pre-exploration filters for one primitive instance.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveInstanceExplorationInput {
    pub(super) build_input: PrimitiveBuildInput,
    pub(super) filters: Vec<CandidateFilter>,
}

impl PrimitiveInstanceExplorationInput {
    /// Creates the exploration input for one primitive instance.
    pub fn new(build_input: PrimitiveBuildInput, filters: Vec<CandidateFilter>) -> Self {
        Self {
            build_input,
            filters,
        }
    }

    /// Returns the values used to build the primitive candidates.
    pub const fn build_input(&self) -> &PrimitiveBuildInput {
        &self.build_input
    }

    /// Returns the filters applied locally after building the candidates.
    pub fn filters(&self) -> &[CandidateFilter] {
        &self.filters
    }
}

/// Precomputed candidates and local filters for one compact submacro instance.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactMacroInstanceExplorationInput {
    pub(super) candidates: CandidateSet,
    pub(super) filters: Vec<CandidateFilter>,
}

impl CompactMacroInstanceExplorationInput {
    /// Creates the exploration input for one compact submacro instance.
    pub fn new(candidates: CandidateSet, filters: Vec<CandidateFilter>) -> Self {
        Self {
            candidates,
            filters,
        }
    }

    /// Returns the candidate set projected by the explored child macro.
    pub const fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }

    /// Returns the filters applied locally to the projected candidates.
    pub fn filters(&self) -> &[CandidateFilter] {
        &self.filters
    }
}

/// Runtime data required to build the candidate sets of one macro.
///
/// Device LUTs are borrowed because they are shared, read-only inputs that may
/// be substantially larger than the per-instance exploration configuration.
#[derive(Clone, Debug, Default)]
pub struct MacroExplorationInput<'lut> {
    pub(super) device_models: HashMap<String, &'lut DeviceLut>,
    pub(super) primitive_instances: HashMap<String, PrimitiveInstanceExplorationInput>,
    pub(super) compact_macro_instances: HashMap<String, CompactMacroInstanceExplorationInput>,
}

impl<'lut> MacroExplorationInput<'lut> {
    /// Creates an empty exploration input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one shared LUT under the device type used by primitive manifests.
    pub fn register_device_model(
        &mut self,
        device_type: impl Into<String>,
        model: &'lut DeviceLut,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let device_type = device_type.into();
        if device_type.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyDeviceType);
        }
        if self.device_models.contains_key(&device_type) {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateDeviceModel { device_type },
            );
        }
        self.device_models.insert(device_type, model);
        Ok(())
    }

    /// Registers build data and filters for one primitive instance path.
    pub fn register_primitive_instance(
        &mut self,
        instance_path: impl Into<String>,
        input: PrimitiveInstanceExplorationInput,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let instance_path = self.validate_new_instance_path(instance_path.into())?;
        self.primitive_instances.insert(instance_path, input);
        Ok(())
    }

    /// Registers projected candidates and filters for one compact submacro path.
    pub fn register_compact_macro_instance(
        &mut self,
        instance_path: impl Into<String>,
        input: CompactMacroInstanceExplorationInput,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let instance_path = self.validate_new_instance_path(instance_path.into())?;
        self.compact_macro_instances.insert(instance_path, input);
        Ok(())
    }

    /// Returns the LUT registered for one primitive device type.
    pub fn device_model(&self, device_type: &str) -> Option<&'lut DeviceLut> {
        self.device_models.get(device_type).copied()
    }

    /// Returns the input registered for one primitive instance path.
    pub fn primitive_instance(
        &self,
        instance_path: &str,
    ) -> Option<&PrimitiveInstanceExplorationInput> {
        self.primitive_instances.get(instance_path)
    }

    /// Returns the compact input registered for one submacro instance path.
    pub fn compact_macro_instance(
        &self,
        instance_path: &str,
    ) -> Option<&CompactMacroInstanceExplorationInput> {
        self.compact_macro_instances.get(instance_path)
    }

    fn validate_new_instance_path(
        &self,
        instance_path: String,
    ) -> Result<String, MacroExplorationInputRegistrationError> {
        if instance_path.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyInstancePath);
        }
        if self.primitive_instances.contains_key(&instance_path)
            || self.compact_macro_instances.contains_key(&instance_path)
        {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateInstanceInput { instance_path },
            );
        }
        Ok(instance_path)
    }
}

/// Errors produced while registering runtime macro exploration inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroExplorationInputRegistrationError {
    EmptyDeviceType,
    EmptyInstancePath,
    DuplicateDeviceModel { device_type: String },
    DuplicateInstanceInput { instance_path: String },
}

impl fmt::Display for MacroExplorationInputRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDeviceType => formatter.write_str("device type cannot be empty"),
            Self::EmptyInstancePath => formatter.write_str("instance path cannot be empty"),
            Self::DuplicateDeviceModel { device_type } => {
                write!(
                    formatter,
                    "device model '{device_type}' is already registered"
                )
            }
            Self::DuplicateInstanceInput { instance_path } => write!(
                formatter,
                "exploration input for instance '{instance_path}' is already registered"
            ),
        }
    }
}

impl Error for MacroExplorationInputRegistrationError {}

/// Kind of circuit instance expected by a macro exploration input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroExplorationInstanceKind {
    Primitive,
    CompactMacro,
    LinearElement,
}

impl fmt::Display for MacroExplorationInstanceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive => formatter.write_str("primitive"),
            Self::CompactMacro => formatter.write_str("compact macro"),
            Self::LinearElement => formatter.write_str("linear element"),
        }
    }
}

/// One invalid or missing item in the runtime exploration input of a macro.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroExplorationInputValidationError {
    UnknownInstance {
        macro_name: String,
        instance_path: String,
    },
    InstanceKindMismatch {
        macro_name: String,
        instance_path: String,
        expected: MacroExplorationInstanceKind,
        actual: MacroExplorationInstanceKind,
    },
    MissingPrimitiveInput {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingCompactMacroInput {
        macro_name: String,
        instance_path: String,
        referenced_macro: String,
    },
    UnknownPrimitive {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingPrimitiveDeviceType {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingDeviceModel {
        macro_name: String,
        instance_path: String,
        primitive: String,
        device_type: String,
    },
}

impl fmt::Display for MacroExplorationInputValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownInstance {
                macro_name,
                instance_path,
            } => write!(
                formatter,
                "macro '{macro_name}' has exploration input for unknown instance '{instance_path}'"
            ),
            Self::InstanceKindMismatch {
                macro_name,
                instance_path,
                expected,
                actual,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' is a {actual}, but its exploration input expects a {expected}"
            ),
            Self::MissingPrimitiveInput {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' primitive instance '{instance_path}' ('{primitive}') has no build input"
            ),
            Self::MissingCompactMacroInput {
                macro_name,
                instance_path,
                referenced_macro,
            } => write!(
                formatter,
                "macro '{macro_name}' submacro instance '{instance_path}' ('{referenced_macro}') has no compact candidate input"
            ),
            Self::UnknownPrimitive {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' references unknown primitive '{primitive}'"
            ),
            Self::MissingPrimitiveDeviceType {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' has no device type"
            ),
            Self::MissingDeviceModel {
                macro_name,
                instance_path,
                primitive,
                device_type,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' requires missing device model '{device_type}'"
            ),
        }
    }
}

impl Error for MacroExplorationInputValidationError {}

/// Validates runtime exploration inputs against the implementation circuit.
///
/// Every independent problem is returned so callers can report a complete,
/// instance-qualified configuration error before candidate construction starts.
pub fn validate_macro_exploration_input(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    input: &MacroExplorationInput<'_>,
) -> Vec<MacroExplorationInputValidationError> {
    let mut errors = Vec::new();

    for instance_path in input.primitive_instances.keys() {
        validate_registered_instance_kind(
            macro_,
            instance_path,
            MacroExplorationInstanceKind::Primitive,
            &mut errors,
        );
    }
    for instance_path in input.compact_macro_instances.keys() {
        validate_registered_instance_kind(
            macro_,
            instance_path,
            MacroExplorationInstanceKind::CompactMacro,
            &mut errors,
        );
    }

    for instance in macro_.circuit().instances() {
        match instance.block() {
            BlockRef::Primitive(primitive) => {
                if !input.primitive_instances.contains_key(instance.name()) {
                    errors.push(
                        MacroExplorationInputValidationError::MissingPrimitiveInput {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            primitive: primitive.clone(),
                        },
                    );
                }

                let Some(manifest) = primitive_catalog.get(primitive) else {
                    errors.push(MacroExplorationInputValidationError::UnknownPrimitive {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance.name().to_owned(),
                        primitive: primitive.clone(),
                    });
                    continue;
                };
                let Some(device_type) = manifest
                    .transistor_type
                    .as_deref()
                    .filter(|device_type| !device_type.trim().is_empty())
                else {
                    errors.push(
                        MacroExplorationInputValidationError::MissingPrimitiveDeviceType {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            primitive: primitive.clone(),
                        },
                    );
                    continue;
                };
                if !input.device_models.contains_key(device_type) {
                    errors.push(MacroExplorationInputValidationError::MissingDeviceModel {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance.name().to_owned(),
                        primitive: primitive.clone(),
                        device_type: device_type.to_owned(),
                    });
                }
            }
            BlockRef::Macro(referenced_macro) => {
                if !input.compact_macro_instances.contains_key(instance.name()) {
                    errors.push(
                        MacroExplorationInputValidationError::MissingCompactMacroInput {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            referenced_macro: referenced_macro.clone(),
                        },
                    );
                }
            }
            BlockRef::Element(_) => {}
        }
    }

    errors
}

fn validate_registered_instance_kind(
    macro_: &Macro,
    instance_path: &str,
    expected: MacroExplorationInstanceKind,
    errors: &mut Vec<MacroExplorationInputValidationError>,
) {
    let Some(instance) = macro_.circuit().instance(instance_path) else {
        errors.push(MacroExplorationInputValidationError::UnknownInstance {
            macro_name: macro_.name().to_owned(),
            instance_path: instance_path.to_owned(),
        });
        return;
    };
    let actual = match instance.block() {
        BlockRef::Primitive(_) => MacroExplorationInstanceKind::Primitive,
        BlockRef::Macro(_) => MacroExplorationInstanceKind::CompactMacro,
        BlockRef::Element(_) => MacroExplorationInstanceKind::LinearElement,
    };
    if actual != expected {
        errors.push(MacroExplorationInputValidationError::InstanceKindMismatch {
            macro_name: macro_.name().to_owned(),
            instance_path: instance_path.to_owned(),
            expected,
            actual,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use shapeic_lut::LookupTable;

    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};

    use super::*;

    fn primitive(name: &str, device_type: Option<&str>) -> PrimitiveManifest {
        PrimitiveManifest {
            name: name.to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: name.to_owned(),
            pins: vec![Pin {
                name: "OUT".to_owned(),
                role: PinRole::Output,
            }],
            files: PrimitiveFiles {
                netlist: "netlist.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            transistor_type: device_type.map(str::to_owned),
            layout_params: None,
            lut_config: None,
            build: None,
        }
    }

    fn macro_with_primitive_and_submacro() -> Macro {
        Macro::new(
            "ota",
            Vec::new(),
            Circuit::builder()
                .primitive("xdp", "diffpair", [("OUT", "VOUT")])
                .macro_instance("xload", "active_load", [("OUT", "VOUT")])
                .resistor("rout", "VOUT", "0", 1.0e6)
                .build(),
            Circuit::builder()
                .resistor("rout", "VOUT", "0", 1.0e6)
                .build(),
        )
    }

    fn one_candidate(name: &str) -> CandidateSet {
        CandidateSet::new(
            name,
            vec![CandidatePoint::new(vec![("gm".to_owned(), 1.0e-3)])],
        )
    }

    fn fixture() -> LookupTable {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shapeic-lut/tests/fixtures/sstadex_combined.npz");
        LookupTable::open(path).expect("LUT fixture should load")
    }

    #[test]
    fn registers_borrowed_models_and_typed_instance_inputs() {
        let table = fixture();
        let nmos = table.model("fixture_nmos").unwrap();
        let mut input = MacroExplorationInput::new();

        input.register_device_model("nmos", nmos).unwrap();
        input
            .register_primitive_instance(
                "xdp",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(one_candidate("active_load"), Vec::new()),
            )
            .unwrap();

        assert_eq!(
            input.device_model("nmos").map(DeviceLut::name),
            Some("fixture_nmos")
        );
        assert!(input.primitive_instance("xdp").is_some());
        assert!(input.compact_macro_instance("xload").is_some());
        assert_eq!(
            input.register_primitive_instance(
                "xload",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            ),
            Err(
                MacroExplorationInputRegistrationError::DuplicateInstanceInput {
                    instance_path: "xload".to_owned(),
                }
            )
        );
    }

    #[test]
    fn accepts_a_complete_input_for_the_implementation_circuit() {
        let table = fixture();
        let nmos = table.model("fixture_nmos").unwrap();
        let macro_ = macro_with_primitive_and_submacro();
        let mut primitives = PrimitiveCatalog::new();
        primitives.register(primitive("diffpair", Some("nmos")));
        let mut input = MacroExplorationInput::new();
        input.register_device_model("nmos", nmos).unwrap();
        input
            .register_primitive_instance(
                "xdp",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(one_candidate("active_load"), Vec::new()),
            )
            .unwrap();

        assert!(validate_macro_exploration_input(&macro_, &primitives, &input).is_empty());
    }

    #[test]
    fn reports_all_missing_unknown_and_mistyped_instance_inputs() {
        let macro_ = macro_with_primitive_and_submacro();
        let mut primitives = PrimitiveCatalog::new();
        primitives.register(primitive("diffpair", Some("nmos")));
        let mut input = MacroExplorationInput::new();
        input
            .register_compact_macro_instance(
                "xdp",
                CompactMacroInstanceExplorationInput::new(one_candidate("wrong"), Vec::new()),
            )
            .unwrap();
        input
            .register_primitive_instance(
                "ghost",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();

        let errors = validate_macro_exploration_input(&macro_, &primitives, &input);

        assert!(errors.contains(
            &MacroExplorationInputValidationError::InstanceKindMismatch {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                expected: MacroExplorationInstanceKind::CompactMacro,
                actual: MacroExplorationInstanceKind::Primitive,
            }
        ));
        assert!(
            errors.contains(&MacroExplorationInputValidationError::UnknownInstance {
                macro_name: "ota".to_owned(),
                instance_path: "ghost".to_owned(),
            })
        );
        assert!(errors.contains(
            &MacroExplorationInputValidationError::MissingPrimitiveInput {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                primitive: "diffpair".to_owned(),
            }
        ));
        assert!(
            errors.contains(&MacroExplorationInputValidationError::MissingDeviceModel {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                primitive: "diffpair".to_owned(),
                device_type: "nmos".to_owned(),
            })
        );
        assert!(errors.contains(
            &MacroExplorationInputValidationError::MissingCompactMacroInput {
                macro_name: "ota".to_owned(),
                instance_path: "xload".to_owned(),
                referenced_macro: "active_load".to_owned(),
            }
        ));
    }
}
