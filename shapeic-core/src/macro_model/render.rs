//! In-memory expansion of macro implementations into small-signal netlists.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::{BlockRef, CircuitInstance, CircuitValue, LinearElement};
use crate::netlist::names::{small_signal_element_name, small_signal_param_name};

use super::{Macro, MacroCatalog, MacroValidationError, validate_macro};

/// Expanded SPICE source and its deterministic numerical parameter order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpandedSmallSignalNetlist {
    source: String,
    parameter_order: Vec<String>,
}

impl ExpandedSmallSignalNetlist {
    /// Returns the linear SPICE source held entirely in memory.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the exact parameter order expected by numerical preparation.
    pub fn parameter_order(&self) -> &[String] {
        &self.parameter_order
    }
}

/// Errors produced while expanding a macro implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroRenderError {
    InvalidMacro {
        macro_name: String,
        errors: Vec<MacroValidationError>,
    },
    MissingPrimitive {
        primitive: String,
    },
    MissingMacro {
        macro_name: String,
    },
    MissingSmallSignalModel {
        primitive: String,
    },
    MissingConnection {
        macro_name: String,
        instance: String,
        port: String,
    },
    DuplicateParameter {
        parameter: String,
    },
    CyclicDependency {
        path: Vec<String>,
    },
}

impl fmt::Display for MacroRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMacro { macro_name, errors } => write!(
                formatter,
                "macro '{macro_name}' failed validation with {} error(s)",
                errors.len()
            ),
            Self::MissingPrimitive { primitive } => {
                write!(formatter, "primitive '{primitive}' is not registered")
            }
            Self::MissingMacro { macro_name } => {
                write!(formatter, "macro '{macro_name}' is not registered")
            }
            Self::MissingSmallSignalModel { primitive } => {
                write!(
                    formatter,
                    "primitive '{primitive}' has no small-signal model"
                )
            }
            Self::MissingConnection {
                macro_name,
                instance,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' has no connection for port '{port}'"
            ),
            Self::DuplicateParameter { parameter } => write!(
                formatter,
                "expanded small-signal parameter '{parameter}' has multiple independent origins"
            ),
            Self::CyclicDependency { path } => {
                write!(formatter, "cyclic macro dependency: {}", path.join(" -> "))
            }
        }
    }
}

impl Error for MacroRenderError {}

/// Expands all reachable primitive branches and linear implementation elements.
///
/// Submacros are recursively expanded. Compact models, testbench composition,
/// and candidate-value binding are intentionally handled by later stages.
pub fn render_expanded_small_signal_netlist(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
) -> Result<ExpandedSmallSignalNetlist, MacroRenderError> {
    let mut renderer = Renderer::default();
    let root_scope = NetScope::root(macro_);
    let mut macro_stack = vec![macro_.name().to_owned()];
    renderer.expand_macro(
        macro_,
        primitive_catalog,
        macro_catalog,
        &root_scope,
        &mut Vec::new(),
        &mut macro_stack,
    )?;

    let source = if renderer.lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", renderer.lines.join("\n"))
    };
    Ok(ExpandedSmallSignalNetlist {
        source,
        parameter_order: renderer.parameter_order,
    })
}

#[derive(Default)]
struct Renderer {
    lines: Vec<String>,
    parameter_order: Vec<String>,
    parameters: HashMap<String, String>,
}

impl Renderer {
    fn expand_macro(
        &mut self,
        macro_: &Macro,
        primitive_catalog: &PrimitiveCatalog,
        macro_catalog: &MacroCatalog,
        scope: &NetScope,
        instance_path: &mut Vec<String>,
        macro_stack: &mut Vec<String>,
    ) -> Result<(), MacroRenderError> {
        let validation_errors = validate_macro(macro_, primitive_catalog, macro_catalog);
        if !validation_errors.is_empty() {
            return Err(MacroRenderError::InvalidMacro {
                macro_name: macro_.name().to_owned(),
                errors: validation_errors,
            });
        }

        for instance in macro_.circuit().instances() {
            instance_path.push(instance.name().to_owned());
            match instance.block() {
                BlockRef::Primitive(primitive_name) => self.expand_primitive(
                    macro_,
                    instance,
                    primitive_name,
                    primitive_catalog,
                    scope,
                    instance_path,
                )?,
                BlockRef::Macro(nested_name) => {
                    if let Some(cycle_start) = macro_stack
                        .iter()
                        .position(|macro_name| macro_name == nested_name)
                    {
                        let mut path = macro_stack[cycle_start..].to_vec();
                        path.push(nested_name.clone());
                        return Err(MacroRenderError::CyclicDependency { path });
                    }
                    let nested = macro_catalog.get(nested_name).ok_or_else(|| {
                        MacroRenderError::MissingMacro {
                            macro_name: nested_name.clone(),
                        }
                    })?;
                    let nested_scope =
                        NetScope::nested(macro_, instance, nested, scope, instance_path)?;
                    macro_stack.push(nested_name.clone());
                    self.expand_macro(
                        nested,
                        primitive_catalog,
                        macro_catalog,
                        &nested_scope,
                        instance_path,
                        macro_stack,
                    )?;
                    macro_stack.pop();
                }
                BlockRef::Element(element) => {
                    self.render_linear_element(macro_, instance, element, scope, instance_path)?;
                }
            }
            instance_path.pop();
        }
        Ok(())
    }

    fn expand_primitive(
        &mut self,
        macro_: &Macro,
        instance: &CircuitInstance,
        primitive_name: &str,
        primitive_catalog: &PrimitiveCatalog,
        scope: &NetScope,
        instance_path: &[String],
    ) -> Result<(), MacroRenderError> {
        let primitive = primitive_catalog.get(primitive_name).ok_or_else(|| {
            MacroRenderError::MissingPrimitive {
                primitive: primitive_name.to_owned(),
            }
        })?;
        let model = primitive.small_signal.as_ref().ok_or_else(|| {
            MacroRenderError::MissingSmallSignalModel {
                primitive: primitive_name.to_owned(),
            }
        })?;
        let path = sanitized_path(instance_path);

        for branch in model.branches() {
            let drain = resolved_connection(macro_, instance, branch.drain_pin(), scope)?;
            let gate = resolved_connection(macro_, instance, branch.gate_pin(), scope)?;
            let source = resolved_connection(macro_, instance, branch.source_pin(), scope)?;
            let gm = small_signal_param_name("gm", &path, branch.name());
            let ro = small_signal_param_name("ro", &path, branch.name());
            self.push_parameter(gm.clone(), format!("primitive:{path}:{}:gm", branch.name()))?;
            self.push_parameter(ro.clone(), format!("primitive:{path}:{}:ro", branch.name()))?;

            self.lines.push(format!(
                "{} {drain} {source} {gate} {source} {gm}",
                small_signal_element_name("G", "gm", &path, branch.name())
            ));
            self.lines.push(format!(
                "{} {drain} {source} {ro}",
                small_signal_element_name("R", "ro", &path, branch.name())
            ));
        }
        Ok(())
    }

    fn render_linear_element(
        &mut self,
        macro_: &Macro,
        instance: &CircuitInstance,
        element: &LinearElement,
        scope: &NetScope,
        instance_path: &[String],
    ) -> Result<(), MacroRenderError> {
        let path = sanitized_path(instance_path);
        let value = element_value(element);
        let rendered_value = match value {
            CircuitValue::Constant(value) => format!("{value:.17e}"),
            CircuitValue::Parameter(parameter) => {
                let rendered = scoped_parameter_name(parameter, &scope.path);
                self.push_parameter(
                    rendered.clone(),
                    format!("macro:{}:{parameter}", scope.path.join("__")),
                )?;
                rendered
            }
        };
        let connection = |port| resolved_connection(macro_, instance, port, scope);

        let line = match element {
            LinearElement::Resistor { .. } => {
                format!(
                    "R_{path} {} {} {rendered_value}",
                    connection("p")?,
                    connection("n")?
                )
            }
            LinearElement::Capacitor { .. } => {
                format!(
                    "C_{path} {} {} {rendered_value}",
                    connection("p")?,
                    connection("n")?
                )
            }
            LinearElement::CurrentSource { .. } => {
                format!(
                    "I_{path} {} {} {rendered_value}",
                    connection("p")?,
                    connection("n")?
                )
            }
            LinearElement::VoltageSource { .. } => {
                format!(
                    "V_{path} {} {} {rendered_value}",
                    connection("p")?,
                    connection("n")?
                )
            }
            LinearElement::VoltageControlledCurrentSource { .. } => format!(
                "G_{path} {} {} {} {} {rendered_value}",
                connection("p")?,
                connection("n")?,
                connection("cp")?,
                connection("cn")?
            ),
        };
        self.lines.push(line);
        Ok(())
    }

    fn push_parameter(
        &mut self,
        parameter: String,
        origin: String,
    ) -> Result<(), MacroRenderError> {
        if let Some(existing_origin) = self.parameters.get(&parameter) {
            if existing_origin == &origin {
                return Ok(());
            }
            return Err(MacroRenderError::DuplicateParameter { parameter });
        }
        self.parameters.insert(parameter.clone(), origin);
        self.parameter_order.push(parameter);
        Ok(())
    }
}

struct NetScope {
    path: Vec<String>,
    public_nets: HashMap<String, String>,
}

impl NetScope {
    fn root(macro_: &Macro) -> Self {
        Self {
            path: Vec::new(),
            public_nets: macro_
                .ports()
                .iter()
                .map(|port| (port.name().to_owned(), port.name().to_owned()))
                .collect(),
        }
    }

    fn nested(
        parent: &Macro,
        instance: &CircuitInstance,
        nested: &Macro,
        parent_scope: &Self,
        instance_path: &[String],
    ) -> Result<Self, MacroRenderError> {
        let mut public_nets = HashMap::new();
        for port in nested.ports() {
            let parent_net = connection(parent, instance, port.name())?;
            public_nets.insert(port.name().to_owned(), parent_scope.resolve(parent_net));
        }
        Ok(Self {
            path: instance_path.to_vec(),
            public_nets,
        })
    }

    fn resolve(&self, net: &str) -> String {
        if let Some(resolved) = self.public_nets.get(net) {
            return resolved.clone();
        }
        if self.path.is_empty() {
            net.to_owned()
        } else {
            format!("n__{}__{}", sanitized_path(&self.path), sanitize(net))
        }
    }
}

fn resolved_connection(
    macro_: &Macro,
    instance: &CircuitInstance,
    port: &str,
    scope: &NetScope,
) -> Result<String, MacroRenderError> {
    Ok(scope.resolve(connection(macro_, instance, port)?))
}

fn connection<'a>(
    macro_: &Macro,
    instance: &'a CircuitInstance,
    port: &str,
) -> Result<&'a str, MacroRenderError> {
    instance
        .net_for_port(port)
        .ok_or_else(|| MacroRenderError::MissingConnection {
            macro_name: macro_.name().to_owned(),
            instance: instance.name().to_owned(),
            port: port.to_owned(),
        })
}

fn element_value(element: &LinearElement) -> &CircuitValue {
    match element {
        LinearElement::Resistor { resistance } => resistance,
        LinearElement::Capacitor { capacitance } => capacitance,
        LinearElement::CurrentSource { current } => current,
        LinearElement::VoltageSource { voltage } => voltage,
        LinearElement::VoltageControlledCurrentSource { transconductance } => transconductance,
    }
}

fn scoped_parameter_name(parameter: &str, scope: &[String]) -> String {
    if scope.is_empty() {
        sanitize(parameter)
    } else {
        format!("{}__{}", sanitize(parameter), sanitized_path(scope))
    }
}

fn sanitized_path(path: &[String]) -> String {
    path.iter()
        .map(|part| sanitize(part))
        .collect::<Vec<_>>()
        .join("__")
}

fn sanitize(value: &str) -> String {
    value.trim().replace('-', "_")
}

#[cfg(test)]
mod tests {
    use crate::catalog::primitive_catalog::PrimitiveCatalog;
    use crate::circuit::Circuit;
    use crate::macro_model::{Macro, MacroCatalog, MacroPort, MacroPortRole};
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
    use crate::primitive::small_signal::{SmallSignalBranch, SmallSignalModel};
    use shapeic_mna::spice2cir::spice2cir_text;

    use super::render_expanded_small_signal_netlist;

    fn primitive_catalog() -> PrimitiveCatalog {
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(PrimitiveManifest {
            name: "gain_primitive".to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: "gain_primitive".to_owned(),
            pins: vec![
                Pin {
                    name: "VIN".to_owned(),
                    role: PinRole::Input,
                },
                Pin {
                    name: "VOUT".to_owned(),
                    role: PinRole::Output,
                },
                Pin {
                    name: "VSS".to_owned(),
                    role: PinRole::Supply,
                },
            ],
            files: PrimitiveFiles {
                netlist: "gain.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: Some(SmallSignalModel::new(vec![SmallSignalBranch::new(
                "m1", "VOUT", "VIN", "VSS",
            )])),
            transistor_type: None,
            layout_params: None,
            lut_config: None,
            build: None,
        });
        catalog
    }

    fn ports() -> Vec<MacroPort> {
        vec![
            MacroPort::new("VIN", MacroPortRole::Input),
            MacroPort::new("VOUT", MacroPortRole::Output),
            MacroPort::new("VSS", MacroPortRole::Ground),
        ]
    }

    fn compact_model() -> Circuit {
        Circuit::builder()
            .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
            .build()
    }

    fn leaf(name: &str, output_net: &str) -> Macro {
        let circuit = Circuit::builder()
            .primitive(
                "xcore",
                "gain_primitive",
                [("VIN", "VIN"), ("VOUT", output_net), ("VSS", "VSS")],
            )
            .resistor("rload", output_net, "VSS", "load_resistance");
        let circuit = if output_net == "VOUT" {
            circuit
        } else {
            circuit.resistor("rout", output_net, "VOUT", 1.0)
        };
        Macro::new(name, ports(), circuit.build(), compact_model())
    }

    #[test]
    fn expands_primitive_branches_and_returns_parameter_order() {
        let macro_ = leaf("leaf", "VOUT");
        let macros = MacroCatalog::from_macros([macro_.clone()]).unwrap();

        let rendered =
            render_expanded_small_signal_netlist(&macro_, &primitive_catalog(), &macros).unwrap();

        assert!(
            rendered
                .source()
                .contains("G_gm__xcore__m1 VOUT VSS VIN VSS gm__xcore__m1")
        );
        assert!(
            rendered
                .source()
                .contains("R_ro__xcore__m1 VOUT VSS ro__xcore__m1")
        );
        assert!(
            rendered
                .source()
                .contains("R_rload VOUT VSS load_resistance")
        );
        assert_eq!(
            rendered.parameter_order(),
            ["gm__xcore__m1", "ro__xcore__m1", "load_resistance"]
        );
        assert!(spice2cir_text(rendered.source()).is_ok());
    }

    #[test]
    fn namespaces_nested_parameters_and_internal_nets_per_instance() {
        let leaf = leaf("leaf", "NINT");
        let parent = Macro::new(
            "parent",
            ports(),
            Circuit::builder()
                .macro_instance(
                    "xa",
                    "leaf",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .macro_instance(
                    "xb",
                    "leaf",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            compact_model(),
        );
        let macros = MacroCatalog::from_macros([leaf, parent.clone()]).unwrap();

        let rendered =
            render_expanded_small_signal_netlist(&parent, &primitive_catalog(), &macros).unwrap();

        assert!(rendered.source().contains("n__xa__NINT"));
        assert!(rendered.source().contains("n__xb__NINT"));
        assert!(
            rendered
                .parameter_order()
                .contains(&"gm__xa__xcore__m1".to_owned())
        );
        assert!(
            rendered
                .parameter_order()
                .contains(&"load_resistance__xb".to_owned())
        );
    }
}
