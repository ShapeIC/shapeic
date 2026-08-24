use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::exploration::candidate::candidate_column_name;
use crate::exploration::combination::{
    CandidateCombinationEquality, CandidateCombinationJoin, CandidateCombinationJoinError,
};

use super::{Macro, MacroCandidateSets, MacroExplorationInstanceKind};

/// Connectivity-derived equalities used to combine macro candidate sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacroCandidateCombinationPlan {
    equalities: Vec<CandidateCombinationEquality>,
}

impl MacroCandidateCombinationPlan {
    /// Returns the cross-instance voltage equalities in deterministic net order.
    pub fn equalities(&self) -> &[CandidateCombinationEquality] {
        &self.equalities
    }
}

/// Lazy traversal over candidate selections compatible with macro connectivity.
#[derive(Debug)]
pub struct MacroCandidateCombinationJoin<'a> {
    plan: MacroCandidateCombinationPlan,
    join: Option<CandidateCombinationJoin<'a>>,
    selection_len: usize,
}

impl<'a> MacroCandidateCombinationJoin<'a> {
    /// Builds the connectivity plan and its indexed lazy join.
    pub fn new(
        macro_: &Macro,
        primitive_catalog: &PrimitiveCatalog,
        candidates: &'a MacroCandidateSets,
    ) -> Result<Self, MacroCandidateCombinationError> {
        let plan = plan_macro_candidate_combinations(macro_, primitive_catalog, candidates)?;
        let candidate_sets = candidates.candidate_sets().collect::<Vec<_>>();
        let selection_len = candidate_sets.len();
        let join = if candidate_sets
            .iter()
            .any(|candidate_set| candidate_set.points.is_empty())
        {
            None
        } else {
            Some(
                CandidateCombinationJoin::new(&candidate_sets, plan.equalities())
                    .map_err(MacroCandidateCombinationError::CandidateJoin)?,
            )
        };
        Ok(Self {
            plan,
            join,
            selection_len,
        })
    }

    /// Returns the connectivity plan used by this traversal.
    pub const fn plan(&self) -> &MacroCandidateCombinationPlan {
        &self.plan
    }

    /// Returns the next reusable instance-to-candidate index selection.
    pub fn next_selection(&mut self) -> Option<&[usize]> {
        self.join
            .as_mut()
            .and_then(CandidateCombinationJoin::next_selection)
    }

    /// Returns the number of instance indices in every yielded selection.
    pub const fn selection_len(&self) -> usize {
        self.selection_len
    }
}

/// Errors produced while deriving or preparing macro candidate combinations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCandidateCombinationError {
    CandidateInstanceNotFound {
        macro_name: String,
        instance_path: String,
    },
    CandidateInstanceKindMismatch {
        macro_name: String,
        instance_path: String,
        expected: MacroExplorationInstanceKind,
        actual: MacroExplorationInstanceKind,
    },
    UnknownPrimitive {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingPrimitiveBuildSpec {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    UnknownPortVoltagePort {
        macro_name: String,
        instance_path: String,
        primitive: String,
        port: String,
    },
    UnknownMacroInterfacePort {
        macro_name: String,
        instance_path: String,
        referenced_macro: String,
        port: String,
    },
    IntraInstanceSharedNet {
        macro_name: String,
        instance_path: String,
        net: String,
        first_port: String,
        second_port: String,
    },
    CandidateJoin(CandidateCombinationJoinError),
}

impl fmt::Display for MacroCandidateCombinationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateInstanceNotFound {
                macro_name,
                instance_path,
            } => write!(
                formatter,
                "macro '{macro_name}' candidate instance '{instance_path}' is not present in its implementation circuit"
            ),
            Self::CandidateInstanceKindMismatch {
                macro_name,
                instance_path,
                expected,
                actual,
            } => write!(
                formatter,
                "macro '{macro_name}' candidate instance '{instance_path}' is a {actual}, expected a {expected}"
            ),
            Self::UnknownPrimitive {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' references unknown primitive '{primitive}' while planning candidate combinations"
            ),
            Self::MissingPrimitiveBuildSpec {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' has no build specification"
            ),
            Self::UnknownPortVoltagePort {
                macro_name,
                instance_path,
                primitive,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' declares port-voltage input '{port}' without a matching circuit connection"
            ),
            Self::UnknownMacroInterfacePort {
                macro_name,
                instance_path,
                referenced_macro,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' projected from '{referenced_macro}' exposes interface port '{port}' without a matching circuit connection"
            ),
            Self::IntraInstanceSharedNet {
                macro_name,
                instance_path,
                net,
                first_port,
                second_port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' connects port-voltage inputs '{first_port}' and '{second_port}' to shared net '{net}', which requires an intra-instance equality"
            ),
            Self::CandidateJoin(error) => write!(
                formatter,
                "could not prepare the indexed macro candidate join: {error}"
            ),
        }
    }
}

impl Error for MacroCandidateCombinationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CandidateJoin(error) => Some(error),
            _ => None,
        }
    }
}

/// Derives candidate-column equalities from primitive port-voltage connectivity.
pub fn plan_macro_candidate_combinations(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    candidates: &MacroCandidateSets,
) -> Result<MacroCandidateCombinationPlan, MacroCandidateCombinationError> {
    let mut ports_by_net = BTreeMap::<String, Vec<PortVoltageColumn>>::new();

    for (set_index, instance_candidates) in candidates.instances().iter().enumerate() {
        let instance_path = instance_candidates.instance_path();
        let instance = macro_.circuit().instance(instance_path).ok_or_else(|| {
            MacroCandidateCombinationError::CandidateInstanceNotFound {
                macro_name: macro_.name().to_owned(),
                instance_path: instance_path.to_owned(),
            }
        })?;
        match instance.block() {
            BlockRef::Primitive(primitive_name) => {
                ensure_instance_kind(
                    macro_,
                    instance_path,
                    MacroExplorationInstanceKind::Primitive,
                    instance_candidates.kind(),
                )?;
                let primitive = primitive_catalog.get(primitive_name).ok_or_else(|| {
                    MacroCandidateCombinationError::UnknownPrimitive {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance_path.to_owned(),
                        primitive: primitive_name.clone(),
                    }
                })?;
                let build = primitive.build.as_ref().ok_or_else(|| {
                    MacroCandidateCombinationError::MissingPrimitiveBuildSpec {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance_path.to_owned(),
                        primitive: primitive_name.clone(),
                    }
                })?;
                for input in build
                    .inputs
                    .iter()
                    .filter(|input| input.source.as_deref() == Some("port_voltage"))
                {
                    let net = instance.net_for_port(&input.name).ok_or_else(|| {
                        MacroCandidateCombinationError::UnknownPortVoltagePort {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance_path.to_owned(),
                            primitive: primitive_name.clone(),
                            port: input.name.clone(),
                        }
                    })?;
                    ports_by_net
                        .entry(net.to_owned())
                        .or_default()
                        .push(PortVoltageColumn {
                            set_index,
                            instance_path: instance_path.to_owned(),
                            port: input.name.clone(),
                            column: candidate_column_name(
                                instance_path,
                                &input.name.to_ascii_lowercase(),
                            ),
                        });
                }
            }
            BlockRef::Macro(referenced_macro) => {
                ensure_instance_kind(
                    macro_,
                    instance_path,
                    MacroExplorationInstanceKind::CompactMacro,
                    instance_candidates.kind(),
                )?;
                for port in instance_candidates.interface_ports() {
                    let net = instance.net_for_port(port).ok_or_else(|| {
                        MacroCandidateCombinationError::UnknownMacroInterfacePort {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance_path.to_owned(),
                            referenced_macro: referenced_macro.clone(),
                            port: port.clone(),
                        }
                    })?;
                    ports_by_net
                        .entry(net.to_owned())
                        .or_default()
                        .push(PortVoltageColumn {
                            set_index,
                            instance_path: instance_path.to_owned(),
                            port: port.clone(),
                            column: candidate_column_name(
                                instance_path,
                                &port.to_ascii_lowercase(),
                            ),
                        });
                }
            }
            BlockRef::Element(_) => {
                return Err(
                    MacroCandidateCombinationError::CandidateInstanceKindMismatch {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance_path.to_owned(),
                        expected: instance_candidates.kind(),
                        actual: MacroExplorationInstanceKind::LinearElement,
                    },
                );
            }
        }
    }

    let mut equalities = Vec::new();
    for (net, ports) in ports_by_net {
        let Some(first) = ports.first() else {
            continue;
        };
        for port in ports.iter().skip(1) {
            if port.set_index == first.set_index {
                return Err(MacroCandidateCombinationError::IntraInstanceSharedNet {
                    macro_name: macro_.name().to_owned(),
                    instance_path: port.instance_path.clone(),
                    net: net.clone(),
                    first_port: first.port.clone(),
                    second_port: port.port.clone(),
                });
            }
            equalities.push(CandidateCombinationEquality::new(
                first.set_index,
                first.column.clone(),
                port.set_index,
                port.column.clone(),
            ));
        }
    }

    Ok(MacroCandidateCombinationPlan { equalities })
}

fn ensure_instance_kind(
    macro_: &Macro,
    instance_path: &str,
    expected: MacroExplorationInstanceKind,
    actual: MacroExplorationInstanceKind,
) -> Result<(), MacroCandidateCombinationError> {
    if expected == actual {
        return Ok(());
    }
    Err(
        MacroCandidateCombinationError::CandidateInstanceKindMismatch {
            macro_name: macro_.name().to_owned(),
            instance_path: instance_path.to_owned(),
            expected,
            actual,
        },
    )
}

#[derive(Clone, Debug)]
struct PortVoltageColumn {
    set_index: usize,
    instance_path: String,
    port: String,
    column: String,
}

#[cfg(test)]
mod tests {
    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilterReport;
    use crate::primitive::build::{
        PrimitiveBuildInputKind, PrimitiveBuildInputSpec, PrimitiveBuildSpec, SweepMode,
    };
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};

    use super::*;
    use crate::macro_model::MacroInstanceCandidateSet;

    fn primitive(name: &str, ports: &[&str]) -> PrimitiveManifest {
        PrimitiveManifest {
            name: name.to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: name.to_owned(),
            pins: ports
                .iter()
                .map(|port| Pin {
                    name: (*port).to_owned(),
                    role: PinRole::Input,
                })
                .collect(),
            files: PrimitiveFiles {
                netlist: "netlist.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            physical_model: None,
            transistor_type: Some("nmos".to_owned()),
            layout_params: None,
            lut_config: None,
            build: Some(PrimitiveBuildSpec {
                inputs: ports
                    .iter()
                    .map(|port| PrimitiveBuildInputSpec {
                        name: (*port).to_owned(),
                        kind: PrimitiveBuildInputKind::Vector,
                        required: true,
                        source: Some("port_voltage".to_owned()),
                    })
                    .collect(),
                sweep_mode: SweepMode::Cartesian,
                derived: Vec::new(),
                lut: Vec::new(),
                columns: Vec::new(),
            }),
        }
    }

    fn point(values: &[(&str, f64)]) -> CandidatePoint {
        CandidatePoint::new(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        )
    }

    fn instance_candidates(name: &str, points: Vec<CandidatePoint>) -> MacroInstanceCandidateSet {
        MacroInstanceCandidateSet {
            instance_path: name.to_owned(),
            kind: MacroExplorationInstanceKind::Primitive,
            candidates: CandidateSet::new(name, points),
            filter_report: CandidateFilterReport::default(),
            compact_provenance: None,
        }
    }

    fn fixture() -> (Macro, PrimitiveCatalog, MacroCandidateSets) {
        let macro_ = Macro::new(
            "top",
            Vec::new(),
            Circuit::builder()
                .primitive("xa", "a", [("PX", "NX"), ("PY", "NY")])
                .primitive("xb", "b", [("PX", "NX"), ("PY", "NY")])
                .primitive("xc", "c", [("PX", "NX")])
                .build(),
            Circuit::builder().resistor("r", "NX", "0", 1.0).build(),
        );
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive("a", &["PX", "PY"]));
        catalog.register(primitive("b", &["PX", "PY"]));
        catalog.register(primitive("c", &["PX"]));
        let candidates = MacroCandidateSets {
            instances: vec![
                instance_candidates(
                    "xa",
                    vec![
                        point(&[("xa.px", 0.0), ("xa.py", 10.0)]),
                        point(&[("xa.px", 1.0), ("xa.py", 20.0)]),
                    ],
                ),
                instance_candidates(
                    "xb",
                    vec![
                        point(&[("xb.px", 0.0), ("xb.py", 10.0)]),
                        point(&[("xb.px", 0.0), ("xb.py", 20.0)]),
                        point(&[("xb.px", 1.0), ("xb.py", 20.0)]),
                    ],
                ),
                instance_candidates(
                    "xc",
                    vec![point(&[("xc.px", 0.0)]), point(&[("xc.px", 1.0)])],
                ),
            ],
        };
        (macro_, catalog, candidates)
    }

    fn two_instance_fixture(second_net: &str) -> (Macro, PrimitiveCatalog, MacroCandidateSets) {
        let macro_ = Macro::new(
            "top",
            Vec::new(),
            Circuit::builder()
                .primitive("xa", "a", [("PX", "NA")])
                .primitive("xb", "b", [("PX", second_net)])
                .build(),
            Circuit::builder().resistor("r", "NA", "0", 1.0).build(),
        );
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive("a", &["PX"]));
        catalog.register(primitive("b", &["PX"]));
        let candidates = MacroCandidateSets {
            instances: vec![
                instance_candidates(
                    "xa",
                    vec![point(&[("xa.px", 0.0)]), point(&[("xa.px", 1.0)])],
                ),
                instance_candidates(
                    "xb",
                    vec![point(&[("xb.px", 0.0)]), point(&[("xb.px", 1.0)])],
                ),
            ],
        };
        (macro_, catalog, candidates)
    }

    #[test]
    fn derives_minimal_equalities_for_multiple_shared_nets_and_instances() {
        let (macro_, catalog, candidates) = fixture();

        let plan = plan_macro_candidate_combinations(&macro_, &catalog, &candidates).unwrap();

        assert_eq!(
            plan.equalities(),
            [
                CandidateCombinationEquality::new(0, "xa.px", 1, "xb.px"),
                CandidateCombinationEquality::new(0, "xa.px", 2, "xc.px"),
                CandidateCombinationEquality::new(0, "xa.py", 1, "xb.py"),
            ]
        );
    }

    #[test]
    fn lazily_yields_only_connectivity_compatible_candidate_indices() {
        let (macro_, catalog, candidates) = fixture();
        let mut join = MacroCandidateCombinationJoin::new(&macro_, &catalog, &candidates).unwrap();

        let mut selections = Vec::new();
        while let Some(selection) = join.next_selection() {
            selections.push(selection.to_vec());
        }

        assert_eq!(join.selection_len(), 3);
        assert_eq!(selections, [vec![0, 0, 0], vec![1, 2, 1]]);
    }

    #[test]
    fn supports_zero_and_one_shared_net_without_reordering_candidates() {
        let (macro_, catalog, candidates) = two_instance_fixture("NB");
        let mut cartesian =
            MacroCandidateCombinationJoin::new(&macro_, &catalog, &candidates).unwrap();
        assert!(cartesian.plan().equalities().is_empty());
        let mut cartesian_selections = Vec::new();
        while let Some(selection) = cartesian.next_selection() {
            cartesian_selections.push(selection.to_vec());
        }
        assert_eq!(
            cartesian_selections,
            [vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
        );

        let (macro_, catalog, candidates) = two_instance_fixture("NA");
        let mut joined =
            MacroCandidateCombinationJoin::new(&macro_, &catalog, &candidates).unwrap();
        assert_eq!(joined.plan().equalities().len(), 1);
        let mut joined_selections = Vec::new();
        while let Some(selection) = joined.next_selection() {
            joined_selections.push(selection.to_vec());
        }
        assert_eq!(joined_selections, [vec![0, 0], vec![1, 1]]);
    }

    #[test]
    fn an_empty_instance_candidate_set_produces_an_exhausted_join() {
        let (macro_, catalog, mut candidates) = fixture();
        candidates.instances[1].candidates.points.clear();

        let mut join = MacroCandidateCombinationJoin::new(&macro_, &catalog, &candidates).unwrap();

        assert_eq!(join.selection_len(), 3);
        assert_eq!(join.next_selection(), None);
    }

    #[test]
    fn rejects_two_port_voltage_inputs_of_one_instance_on_the_same_net() {
        let macro_ = Macro::new(
            "top",
            Vec::new(),
            Circuit::builder()
                .primitive("xa", "a", [("PX", "N"), ("PY", "N")])
                .build(),
            Circuit::builder().resistor("r", "N", "0", 1.0).build(),
        );
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive("a", &["PX", "PY"]));
        let candidates = MacroCandidateSets {
            instances: vec![instance_candidates(
                "xa",
                vec![point(&[("xa.px", 0.0), ("xa.py", 0.0)])],
            )],
        };

        assert!(matches!(
            plan_macro_candidate_combinations(&macro_, &catalog, &candidates),
            Err(MacroCandidateCombinationError::IntraInstanceSharedNet {
                instance_path,
                net,
                first_port,
                second_port,
                ..
            }) if instance_path == "xa"
                && net == "N"
                && first_port == "PX"
                && second_port == "PY"
        ));
    }
}
