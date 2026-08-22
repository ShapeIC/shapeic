//! Typed circuit topology used by macros and compact models.
//!
//! A circuit is a collection of named instances. Every instance exposes a set
//! of ports connected to circuit nets. Instances may reference primitives,
//! macros, or the linear elements that can be used to describe compact models.

/// A typed circuit made of block instances and their net connections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    instances: Vec<CircuitInstance>,
}

impl Circuit {
    /// Starts building an empty circuit.
    pub fn builder() -> CircuitBuilder {
        CircuitBuilder::default()
    }

    /// Returns the circuit instances in declaration order.
    pub fn instances(&self) -> &[CircuitInstance] {
        &self.instances
    }

    /// Finds an instance by its local name.
    pub fn instance(&self, name: &str) -> Option<&CircuitInstance> {
        self.instances.iter().find(|instance| instance.name == name)
    }
}

/// Builder for the common, concise circuit-construction path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitBuilder {
    instances: Vec<CircuitInstance>,
}

impl CircuitBuilder {
    /// Adds a primitive instance and its port-to-net connections.
    pub fn primitive<I, P, N>(
        mut self,
        name: impl Into<String>,
        primitive: impl Into<String>,
        connections: I,
    ) -> Self
    where
        I: IntoIterator<Item = (P, N)>,
        P: Into<String>,
        N: Into<String>,
    {
        self.instances
            .push(CircuitInstance::primitive(name, primitive, connections));
        self
    }

    /// Adds a submacro instance and its port-to-net connections.
    pub fn macro_instance<I, P, N>(
        mut self,
        name: impl Into<String>,
        macro_name: impl Into<String>,
        connections: I,
    ) -> Self
    where
        I: IntoIterator<Item = (P, N)>,
        P: Into<String>,
        N: Into<String>,
    {
        self.instances.push(CircuitInstance::macro_instance(
            name,
            macro_name,
            connections,
        ));
        self
    }

    /// Adds a resistor connected between `positive` and `negative`.
    pub fn resistor(
        mut self,
        name: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
        resistance: impl Into<CircuitValue>,
    ) -> Self {
        self.instances.push(CircuitInstance::two_terminal_element(
            name,
            LinearElement::Resistor {
                resistance: resistance.into(),
            },
            positive,
            negative,
        ));
        self
    }

    /// Adds a capacitor connected between `positive` and `negative`.
    pub fn capacitor(
        mut self,
        name: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
        capacitance: impl Into<CircuitValue>,
    ) -> Self {
        self.instances.push(CircuitInstance::two_terminal_element(
            name,
            LinearElement::Capacitor {
                capacitance: capacitance.into(),
            },
            positive,
            negative,
        ));
        self
    }

    /// Adds an independent current source from `positive` to `negative`.
    pub fn current_source(
        mut self,
        name: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
        current: impl Into<CircuitValue>,
    ) -> Self {
        self.instances.push(CircuitInstance::two_terminal_element(
            name,
            LinearElement::CurrentSource {
                current: current.into(),
            },
            positive,
            negative,
        ));
        self
    }

    /// Adds an independent voltage source from `positive` to `negative`.
    pub fn voltage_source(
        mut self,
        name: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
        voltage: impl Into<CircuitValue>,
    ) -> Self {
        self.instances.push(CircuitInstance::two_terminal_element(
            name,
            LinearElement::VoltageSource {
                voltage: voltage.into(),
            },
            positive,
            negative,
        ));
        self
    }

    /// Adds a voltage-controlled current source.
    pub fn vccs(
        mut self,
        name: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
        control_positive: impl Into<String>,
        control_negative: impl Into<String>,
        transconductance: impl Into<CircuitValue>,
    ) -> Self {
        self.instances.push(CircuitInstance::new(
            name,
            BlockRef::Element(LinearElement::VoltageControlledCurrentSource {
                transconductance: transconductance.into(),
            }),
            [
                ("p", positive.into()),
                ("n", negative.into()),
                ("cp", control_positive.into()),
                ("cn", control_negative.into()),
            ],
        ));
        self
    }

    /// Finishes construction without performing catalog-dependent validation.
    pub fn build(self) -> Circuit {
        Circuit {
            instances: self.instances,
        }
    }
}

/// One named block placed in a circuit.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitInstance {
    name: String,
    block: BlockRef,
    connections: Vec<PortConnection>,
}

impl CircuitInstance {
    /// Creates an instance from an explicit block reference and connections.
    pub fn new<I, P, N>(name: impl Into<String>, block: BlockRef, connections: I) -> Self
    where
        I: IntoIterator<Item = (P, N)>,
        P: Into<String>,
        N: Into<String>,
    {
        Self {
            name: name.into(),
            block,
            connections: connections
                .into_iter()
                .map(|(port, net)| PortConnection::new(port, net))
                .collect(),
        }
    }

    /// Creates a primitive instance.
    pub fn primitive<I, P, N>(
        name: impl Into<String>,
        primitive: impl Into<String>,
        connections: I,
    ) -> Self
    where
        I: IntoIterator<Item = (P, N)>,
        P: Into<String>,
        N: Into<String>,
    {
        Self::new(name, BlockRef::Primitive(primitive.into()), connections)
    }

    /// Creates a submacro instance.
    pub fn macro_instance<I, P, N>(
        name: impl Into<String>,
        macro_name: impl Into<String>,
        connections: I,
    ) -> Self
    where
        I: IntoIterator<Item = (P, N)>,
        P: Into<String>,
        N: Into<String>,
    {
        Self::new(name, BlockRef::Macro(macro_name.into()), connections)
    }

    fn two_terminal_element(
        name: impl Into<String>,
        element: LinearElement,
        positive: impl Into<String>,
        negative: impl Into<String>,
    ) -> Self {
        Self::new(
            name,
            BlockRef::Element(element),
            [("p", positive.into()), ("n", negative.into())],
        )
    }

    /// Returns the local instance name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the referenced block.
    pub const fn block(&self) -> &BlockRef {
        &self.block
    }

    /// Returns the instance's port-to-net connections.
    pub fn connections(&self) -> &[PortConnection] {
        &self.connections
    }

    /// Returns the net connected to one port.
    pub fn net_for_port(&self, port: &str) -> Option<&str> {
        self.connections
            .iter()
            .find_map(|connection| (connection.port == port).then_some(connection.net.as_str()))
    }
}

/// A circuit block that can be instantiated.
#[derive(Clone, Debug, PartialEq)]
pub enum BlockRef {
    /// A primitive resolved through the primitive catalog.
    Primitive(String),
    /// A macro resolved through the future macro catalog.
    Macro(String),
    /// A built-in linear circuit element.
    Element(LinearElement),
}

/// Linear elements initially supported by typed circuits and compact models.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearElement {
    Resistor { resistance: CircuitValue },
    Capacitor { capacitance: CircuitValue },
    CurrentSource { current: CircuitValue },
    VoltageSource { voltage: CircuitValue },
    VoltageControlledCurrentSource { transconductance: CircuitValue },
}

/// A constant or symbolic value used by a linear circuit element.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitValue {
    Constant(f64),
    Parameter(String),
}

impl CircuitValue {
    /// Creates a symbolic circuit parameter.
    pub fn parameter(name: impl Into<String>) -> Self {
        Self::Parameter(name.into())
    }
}

impl From<f64> for CircuitValue {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl From<String> for CircuitValue {
    fn from(value: String) -> Self {
        Self::Parameter(value)
    }
}

impl From<&str> for CircuitValue {
    fn from(value: &str) -> Self {
        Self::Parameter(value.to_owned())
    }
}

/// One port of an instance connected to a circuit net.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortConnection {
    port: String,
    net: String,
}

impl PortConnection {
    pub fn new(port: impl Into<String>, net: impl Into<String>) -> Self {
        Self {
            port: port.into(),
            net: net.into(),
        }
    }

    /// Returns the local port name.
    pub fn port(&self) -> &str {
        &self.port
    }

    /// Returns the circuit net connected to the port.
    pub fn net(&self) -> &str {
        &self.net
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockRef, Circuit, CircuitValue, LinearElement};

    #[test]
    fn builds_the_ota_topology_from_primitive_instances() {
        let circuit = Circuit::builder()
            .primitive(
                "xdp",
                "simplediffpair",
                [
                    ("VINP", "VINP"),
                    ("VINN", "VINN"),
                    ("VOUTP", "VOUT"),
                    ("VOUTN", "N1"),
                    ("VTAIL", "IBIAS"),
                ],
            )
            .primitive(
                "xcm",
                "simplecurrentmirror",
                [("VINP", "N1"), ("VOUTP", "VOUT"), ("VDD", "VDD")],
            )
            .build();

        assert_eq!(circuit.instances().len(), 2);
        assert_eq!(
            circuit.instance("xdp").map(|instance| instance.block()),
            Some(&BlockRef::Primitive("simplediffpair".to_owned()))
        );
        assert_eq!(
            circuit
                .instance("xdp")
                .and_then(|instance| instance.net_for_port("VOUTN")),
            Some("N1")
        );
        assert_eq!(
            circuit
                .instance("xcm")
                .and_then(|instance| instance.net_for_port("VINP")),
            Some("N1")
        );
    }

    #[test]
    fn builds_a_typed_compact_transconductance_model() {
        let model = Circuit::builder()
            .vccs("gm", "VOUT", "VSS", "VINP", "VINN", "gm_eq")
            .resistor("rout", "VOUT", "VSS", "ro_eq")
            .capacitor("cout", "VOUT", "VSS", 2.0e-12)
            .build();

        assert_eq!(model.instances().len(), 3);
        assert_eq!(
            model.instance("gm").map(|instance| instance.block()),
            Some(&BlockRef::Element(
                LinearElement::VoltageControlledCurrentSource {
                    transconductance: CircuitValue::Parameter("gm_eq".to_owned()),
                }
            ))
        );
        assert_eq!(
            model
                .instance("gm")
                .and_then(|instance| instance.net_for_port("cp")),
            Some("VINP")
        );
        assert_eq!(
            model.instance("cout").map(|instance| instance.block()),
            Some(&BlockRef::Element(LinearElement::Capacitor {
                capacitance: CircuitValue::Constant(2.0e-12),
            }))
        );
    }

    #[test]
    fn represents_submacros_and_independent_sources() {
        let circuit = Circuit::builder()
            .macro_instance("xstage", "gain_stage", [("VIN", "VIN"), ("VOUT", "VOUT")])
            .current_source("ibias", "VDD", "IBIAS", "bias_current")
            .voltage_source("vdd", "VDD", "VSS", 1.5)
            .build();

        assert_eq!(
            circuit.instance("xstage").map(|instance| instance.block()),
            Some(&BlockRef::Macro("gain_stage".to_owned()))
        );
        assert_eq!(
            circuit.instance("ibias").map(|instance| instance.block()),
            Some(&BlockRef::Element(LinearElement::CurrentSource {
                current: CircuitValue::Parameter("bias_current".to_owned()),
            }))
        );
        assert_eq!(
            circuit.instance("vdd").map(|instance| instance.block()),
            Some(&BlockRef::Element(LinearElement::VoltageSource {
                voltage: CircuitValue::Constant(1.5),
            }))
        );
    }
}
