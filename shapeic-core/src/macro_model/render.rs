//! In-memory expansion of macro implementations into small-signal netlists.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::{BlockRef, CircuitInstance, CircuitValue, LinearElement};
use crate::netlist::names::{small_signal_element_name, small_signal_param_name};

use super::{Macro, MacroCatalog, MacroValidationError, validate_macro};

/// Resolved identity and terminal topology of an expanded primitive branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPrimitiveBranch {
    instance_path: String,
    primitive_name: String,
    branch_name: String,
    gate_node: String,
    drain_node: String,
    source_node: String,
}

impl ResolvedPrimitiveBranch {
    /// Returns the sanitized hierarchical path of the primitive instance.
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// Returns the primitive definition referenced by the instance.
    pub fn primitive_name(&self) -> &str {
        &self.primitive_name
    }

    /// Returns the primitive small-signal branch name.
    pub fn branch_name(&self) -> &str {
        &self.branch_name
    }

    /// Returns the gate node resolved in the expanded netlist namespace.
    pub fn gate_node(&self) -> &str {
        &self.gate_node
    }

    /// Returns the drain node resolved in the expanded netlist namespace.
    pub fn drain_node(&self) -> &str {
        &self.drain_node
    }

    /// Returns the source node resolved in the expanded netlist namespace.
    pub fn source_node(&self) -> &str {
        &self.source_node
    }
}

/// Expanded SPICE source, parameter order, and resolved primitive topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpandedSmallSignalNetlist {
    source: String,
    parameter_order: Vec<String>,
    primitive_branches: Vec<ResolvedPrimitiveBranch>,
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

    /// Returns the primitive branches materialized by the selected render mode.
    pub fn primitive_branches(&self) -> &[ResolvedPrimitiveBranch] {
        &self.primitive_branches
    }

    pub(crate) fn into_parts(self) -> (String, Vec<String>) {
        (self.source, self.parameter_order)
    }
}

/// Selects which macro representation is lowered into the linear netlist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MacroRenderMode {
    /// Expands the implementation of the root macro and every submacro.
    #[default]
    Expanded,
    /// Expands the root implementation and uses compact models for submacros.
    CompactSubmacros,
    /// Renders only the compact model of the root macro.
    Compact,
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
                "small-signal parameter '{parameter}' has multiple independent origins"
            ),
            Self::CyclicDependency { path } => {
                write!(formatter, "cyclic macro dependency: {}", path.join(" -> "))
            }
        }
    }
}

impl Error for MacroRenderError {}

/// Fully expands all reachable primitive branches and implementation elements.
pub fn render_expanded_small_signal_netlist(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
) -> Result<ExpandedSmallSignalNetlist, MacroRenderError> {
    render_small_signal_netlist(
        macro_,
        primitive_catalog,
        macro_catalog,
        MacroRenderMode::Expanded,
    )
}

/// Lowers the selected macro representation to an in-memory linear netlist.
///
/// Testbench composition and candidate-value binding are intentionally handled
/// by later preparation stages.
pub fn render_small_signal_netlist(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    mode: MacroRenderMode,
) -> Result<ExpandedSmallSignalNetlist, MacroRenderError> {
    let mut renderer = Renderer::default();
    let root_scope = NetScope::root(macro_);
    match mode {
        MacroRenderMode::Compact => renderer.render_compact_model(
            macro_,
            primitive_catalog,
            macro_catalog,
            &root_scope,
            &mut Vec::new(),
        )?,
        MacroRenderMode::Expanded | MacroRenderMode::CompactSubmacros => {
            renderer.expand_macro(
                macro_,
                primitive_catalog,
                macro_catalog,
                mode,
                &root_scope,
                &mut Vec::new(),
                &mut vec![macro_.name().to_owned()],
            )?;
        }
    }

    let source = if renderer.lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", renderer.lines.join("\n"))
    };
    Ok(ExpandedSmallSignalNetlist {
        source,
        parameter_order: renderer.parameter_order,
        primitive_branches: renderer.primitive_branches,
    })
}

#[derive(Default)]
struct Renderer {
    lines: Vec<String>,
    parameter_order: Vec<String>,
    parameters: HashMap<String, String>,
    primitive_branches: Vec<ResolvedPrimitiveBranch>,
}

impl Renderer {
    fn expand_macro(
        &mut self,
        macro_: &Macro,
        primitive_catalog: &PrimitiveCatalog,
        macro_catalog: &MacroCatalog,
        mode: MacroRenderMode,
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
                    match mode {
                        MacroRenderMode::Expanded => {
                            macro_stack.push(nested_name.clone());
                            self.expand_macro(
                                nested,
                                primitive_catalog,
                                macro_catalog,
                                mode,
                                &nested_scope,
                                instance_path,
                                macro_stack,
                            )?;
                            macro_stack.pop();
                        }
                        MacroRenderMode::CompactSubmacros | MacroRenderMode::Compact => {
                            self.render_compact_model(
                                nested,
                                primitive_catalog,
                                macro_catalog,
                                &nested_scope,
                                instance_path,
                            )?;
                        }
                    }
                }
                BlockRef::Element(element) => {
                    self.render_linear_element(macro_, instance, element, scope, instance_path)?;
                }
            }
            instance_path.pop();
        }
        Ok(())
    }

    fn render_compact_model(
        &mut self,
        macro_: &Macro,
        primitive_catalog: &PrimitiveCatalog,
        macro_catalog: &MacroCatalog,
        scope: &NetScope,
        instance_path: &mut Vec<String>,
    ) -> Result<(), MacroRenderError> {
        let validation_errors = validate_macro(macro_, primitive_catalog, macro_catalog);
        if !validation_errors.is_empty() {
            return Err(MacroRenderError::InvalidMacro {
                macro_name: macro_.name().to_owned(),
                errors: validation_errors,
            });
        }

        for instance in macro_.compact_model().instances() {
            instance_path.push(instance.name().to_owned());
            let BlockRef::Element(element) = instance.block() else {
                unreachable!("validated compact models contain only linear elements");
            };
            self.render_linear_element(macro_, instance, element, scope, instance_path)?;
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
            self.primitive_branches.push(ResolvedPrimitiveBranch {
                instance_path: path.clone(),
                primitive_name: primitive_name.to_owned(),
                branch_name: branch.name().to_owned(),
                gate_node: gate,
                drain_node: drain,
                source_node: source,
            });
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

    use super::{
        MacroRenderMode, render_expanded_small_signal_netlist, render_small_signal_netlist,
    };

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
        let [branch] = rendered.primitive_branches() else {
            panic!("expected one expanded primitive branch");
        };
        assert_eq!(branch.instance_path(), "xcore");
        assert_eq!(branch.primitive_name(), "gain_primitive");
        assert_eq!(branch.branch_name(), "m1");
        assert_eq!(branch.gate_node(), "VIN");
        assert_eq!(branch.drain_node(), "VOUT");
        assert_eq!(branch.source_node(), "VSS");
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
        assert_eq!(rendered.primitive_branches().len(), 2);
        assert_eq!(
            rendered.primitive_branches()[0].instance_path(),
            "xa__xcore"
        );
        assert_eq!(rendered.primitive_branches()[0].drain_node(), "n__xa__NINT");
        assert_eq!(
            rendered.primitive_branches()[1].instance_path(),
            "xb__xcore"
        );
        assert_eq!(rendered.primitive_branches()[1].drain_node(), "n__xb__NINT");
    }

    #[test]
    fn renders_the_root_compact_model_without_expanding_its_implementation() {
        let macro_ = leaf("leaf", "VOUT");
        let macros = MacroCatalog::from_macros([macro_.clone()]).unwrap();

        let rendered = render_small_signal_netlist(
            &macro_,
            &primitive_catalog(),
            &macros,
            MacroRenderMode::Compact,
        )
        .unwrap();

        assert_eq!(rendered.source(), "G_gm VOUT VSS VIN VSS gm_eq\n");
        assert_eq!(rendered.parameter_order(), ["gm_eq"]);
        assert!(rendered.primitive_branches().is_empty());
        assert!(!rendered.source().contains("xcore"));
    }

    #[test]
    fn uses_independent_compact_parameters_for_each_submacro_instance() {
        let leaf = leaf("leaf", "VOUT");
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

        let rendered = render_small_signal_netlist(
            &parent,
            &primitive_catalog(),
            &macros,
            MacroRenderMode::CompactSubmacros,
        )
        .unwrap();

        assert!(
            rendered
                .source()
                .contains("G_xa__gm VOUT VSS VIN VSS gm_eq__xa")
        );
        assert!(
            rendered
                .source()
                .contains("G_xb__gm VOUT VSS VIN VSS gm_eq__xb")
        );
        assert_eq!(rendered.parameter_order(), ["gm_eq__xa", "gm_eq__xb"]);
        assert!(rendered.primitive_branches().is_empty());
        assert!(!rendered.source().contains("gm__xa__xcore__m1"));
        assert!(spice2cir_text(rendered.source()).is_ok());
    }
}
