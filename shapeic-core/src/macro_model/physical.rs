//! Candidate binding and cached physical-LUT stamps for macro exploration.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use shapeic_layout::{
    LayoutAwareAdmittance, LayoutAwarePoint, LayoutError, PhysicalLookupTable, PhysicalPrimitive,
};
use shapeic_mna::numeric::{NumericMnaError, NumericMnaSystem};

use crate::exploration::binding::CandidateParameterBindingError;
use crate::exploration::candidate::{CandidateSet, candidate_column_indices};

use super::ResolvedPhysicalPrimitive;

const PHYSICAL_POINT_PARAMETER_COUNT: usize = 6;

/// Result of attempting to stamp one selected layout-aware candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhysicalCandidateStampOutcome {
    /// Every physical primitive was queried and stamped successfully.
    Stamped,
    /// At least one local primitive candidate lies outside the physical LUT.
    OutOfDomain,
}

/// Resolved, reusable physical bindings for one macro testbench.
#[derive(Debug)]
pub(crate) struct CandidatePhysicalBinder<'a> {
    primitives: Vec<BoundPhysicalPrimitive<'a>>,
    candidate_set_count: usize,
}

impl<'a> CandidatePhysicalBinder<'a> {
    /// Resolves physical LUT primitives, candidate columns, and MNA connections.
    pub(crate) fn new(
        plans: &[ResolvedPhysicalPrimitive],
        candidate_sets: &[&'a CandidateSet],
        physical_lut: &'a PhysicalLookupTable,
    ) -> Result<Self, CandidatePhysicalBindingError> {
        let mut primitives = Vec::with_capacity(plans.len());
        for plan in plans {
            let set_index = candidate_sets
                .iter()
                .position(|candidates| candidates.name == plan.instance_path())
                .ok_or_else(|| CandidatePhysicalBindingError::MissingCandidateSet {
                    instance_path: plan.instance_path().to_owned(),
                })?;
            let columns = plan.candidate_columns();
            let parameter_names = [
                columns.length(),
                columns.finger_width(),
                columns.nf(),
                columns.vbs(),
                columns.vgs(),
                columns.vds(),
            ]
            .map(str::to_owned);
            let candidate_columns = candidate_column_indices(candidate_sets[set_index])
                .map_err(CandidateParameterBindingError::from)?;
            let mut column_indices = [0; PHYSICAL_POINT_PARAMETER_COUNT];
            for (index, parameter) in parameter_names.iter().enumerate() {
                column_indices[index] =
                    candidate_columns.get(parameter).copied().ok_or_else(|| {
                        CandidatePhysicalBindingError::MissingCandidateColumn {
                            instance_path: plan.instance_path().to_owned(),
                            column: parameter.clone(),
                        }
                    })?;
            }
            let primitive = physical_lut
                .primitive(plan.lut_primitive())
                .map_err(|source| CandidatePhysicalBindingError::PhysicalLut {
                    instance_path: plan.instance_path().to_owned(),
                    source,
                })?;
            let connections = resolve_connections(plan, primitive)?;
            let cache = (0..candidate_sets[set_index].points.len())
                .map(|_| None)
                .collect();
            primitives.push(BoundPhysicalPrimitive {
                instance_path: plan.instance_path().to_owned(),
                primitive,
                candidates: candidate_sets[set_index],
                column_indices,
                set_index,
                connections,
                cache,
            });
        }
        Ok(Self {
            primitives,
            candidate_set_count: candidate_sets.len(),
        })
    }

    /// Queries each selected local candidate at most once and stamps its
    /// physical contribution into the supplied numerical MNA system.
    pub(crate) fn stamp(
        &mut self,
        system: &mut NumericMnaSystem,
        candidate_indices: &[usize],
    ) -> Result<PhysicalCandidateStampOutcome, CandidatePhysicalBindingError> {
        let outcome = self.prepare(candidate_indices)?;
        if outcome == PhysicalCandidateStampOutcome::OutOfDomain {
            return Ok(outcome);
        }
        for primitive in &self.primitives {
            let local_index = candidate_indices[primitive.set_index];
            let CachedPhysicalQuery::Admittance(admittance) = primitive.cache[local_index]
                .as_ref()
                .expect("physical query cache entry was initialized")
            else {
                unreachable!("a prepared in-domain physical query must contain admittance data");
            };
            let capacitance = admittance.total_capacitance();
            system
                .stamp_port_admittance(
                    &admittance.ports,
                    &primitive.connections,
                    admittance.interconnect_conductance.view(),
                    capacitance.view(),
                )
                .map_err(|source| CandidatePhysicalBindingError::Stamp {
                    instance_path: primitive.instance_path.clone(),
                    source,
                })?;
        }
        Ok(PhysicalCandidateStampOutcome::Stamped)
    }

    /// Resolves and caches the selected physical queries without stamping MNA.
    pub(crate) fn prepare(
        &mut self,
        candidate_indices: &[usize],
    ) -> Result<PhysicalCandidateStampOutcome, CandidatePhysicalBindingError> {
        if candidate_indices.len() != self.candidate_set_count {
            return Err(CandidatePhysicalBindingError::SelectionCountMismatch {
                expected: self.candidate_set_count,
                actual: candidate_indices.len(),
            });
        }
        for primitive in &mut self.primitives {
            let local_index = candidate_indices[primitive.set_index];
            if local_index >= primitive.cache.len() {
                return Err(CandidatePhysicalBindingError::CandidateIndexOutOfBounds {
                    instance_path: primitive.instance_path.clone(),
                    index: local_index,
                    candidate_count: primitive.cache.len(),
                });
            }

            if primitive.cache[local_index].is_none() {
                let point = &primitive.candidates.points[local_index];
                let values = primitive
                    .column_indices
                    .map(|column_index| point.values[column_index].1);
                let Some(nf) = valid_finger_count(values[2]) else {
                    primitive.cache[local_index] = Some(CachedPhysicalQuery::OutOfDomain);
                    return Ok(PhysicalCandidateStampOutcome::OutOfDomain);
                };
                let point = LayoutAwarePoint::new(
                    values[0], values[1], nf, values[3], values[4], values[5],
                );
                primitive.cache[local_index] =
                    Some(match primitive.primitive.query_layout_aware(point) {
                        Ok(admittance) => CachedPhysicalQuery::Admittance(Box::new(admittance)),
                        Err(
                            LayoutError::OutOfPhysicalRange { .. }
                            | LayoutError::InvalidFingerCount,
                        ) => CachedPhysicalQuery::OutOfDomain,
                        Err(source) => {
                            return Err(CandidatePhysicalBindingError::PhysicalLut {
                                instance_path: primitive.instance_path.clone(),
                                source,
                            });
                        }
                    });
            }

            if matches!(
                primitive.cache[local_index]
                    .as_ref()
                    .expect("physical query cache entry was initialized"),
                CachedPhysicalQuery::OutOfDomain
            ) {
                return Ok(PhysicalCandidateStampOutcome::OutOfDomain);
            }
        }
        Ok(PhysicalCandidateStampOutcome::Stamped)
    }

    #[cfg(test)]
    fn cached_query_count(&self) -> usize {
        self.primitives
            .iter()
            .flat_map(|primitive| &primitive.cache)
            .filter(|entry| entry.is_some())
            .count()
    }
}

#[derive(Debug)]
struct BoundPhysicalPrimitive<'a> {
    instance_path: String,
    primitive: &'a PhysicalPrimitive,
    candidates: &'a CandidateSet,
    column_indices: [usize; PHYSICAL_POINT_PARAMETER_COUNT],
    set_index: usize,
    connections: BTreeMap<String, String>,
    cache: Vec<Option<CachedPhysicalQuery>>,
}

#[derive(Clone, Debug)]
enum CachedPhysicalQuery {
    Admittance(Box<LayoutAwareAdmittance>),
    OutOfDomain,
}

fn resolve_connections(
    plan: &ResolvedPhysicalPrimitive,
    primitive: &PhysicalPrimitive,
) -> Result<BTreeMap<String, String>, CandidatePhysicalBindingError> {
    let connections = plan
        .ports()
        .iter()
        .map(|port| (port.physical_port().to_owned(), port.node().to_owned()))
        .collect::<BTreeMap<_, _>>();
    let expected = primitive.ports();
    if connections.len() != expected.len()
        || expected.iter().any(|port| !connections.contains_key(port))
    {
        return Err(CandidatePhysicalBindingError::IncompatiblePorts {
            instance_path: plan.instance_path().to_owned(),
            physical_primitive: plan.lut_primitive().to_owned(),
            lut_ports: expected.to_vec(),
            descriptor_ports: connections.keys().cloned().collect(),
        });
    }
    Ok(connections)
}

fn valid_finger_count(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 1.0 || value > f64::from(u32::MAX) || value.fract() != 0.0 {
        return None;
    }
    Some(value as u32)
}

/// Errors produced while preparing, querying, or stamping physical candidates.
#[derive(Debug)]
pub enum CandidatePhysicalBindingError {
    MissingCandidateSet {
        instance_path: String,
    },
    MissingCandidateColumn {
        instance_path: String,
        column: String,
    },
    IncompatiblePorts {
        instance_path: String,
        physical_primitive: String,
        lut_ports: Vec<String>,
        descriptor_ports: Vec<String>,
    },
    SelectionCountMismatch {
        expected: usize,
        actual: usize,
    },
    CandidateIndexOutOfBounds {
        instance_path: String,
        index: usize,
        candidate_count: usize,
    },
    CandidateBinding(CandidateParameterBindingError),
    PhysicalLut {
        instance_path: String,
        source: LayoutError,
    },
    Stamp {
        instance_path: String,
        source: NumericMnaError,
    },
}

impl From<CandidateParameterBindingError> for CandidatePhysicalBindingError {
    fn from(error: CandidateParameterBindingError) -> Self {
        Self::CandidateBinding(error)
    }
}

impl fmt::Display for CandidatePhysicalBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCandidateSet { instance_path } => write!(
                formatter,
                "physical primitive instance '{instance_path}' has no local candidate set"
            ),
            Self::MissingCandidateColumn {
                instance_path,
                column,
            } => write!(
                formatter,
                "physical primitive instance '{instance_path}' is missing candidate column '{column}'"
            ),
            Self::IncompatiblePorts {
                instance_path,
                physical_primitive,
                lut_ports,
                descriptor_ports,
            } => write!(
                formatter,
                "physical primitive instance '{instance_path}' maps ports {descriptor_ports:?}, but LUT primitive '{physical_primitive}' requires {lut_ports:?}"
            ),
            Self::SelectionCountMismatch { expected, actual } => write!(
                formatter,
                "physical binding expected at least {expected} candidate indices, received {actual}"
            ),
            Self::CandidateIndexOutOfBounds {
                instance_path,
                index,
                candidate_count,
            } => write!(
                formatter,
                "candidate index {index} is outside physical primitive instance '{instance_path}' with {candidate_count} local candidates"
            ),
            Self::CandidateBinding(error) => {
                write!(formatter, "could not bind physical query columns: {error}")
            }
            Self::PhysicalLut {
                instance_path,
                source,
            } => write!(
                formatter,
                "could not query physical primitive instance '{instance_path}': {source}"
            ),
            Self::Stamp {
                instance_path,
                source,
            } => write!(
                formatter,
                "could not stamp physical primitive instance '{instance_path}': {source}"
            ),
        }
    }
}

impl Error for CandidatePhysicalBindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CandidateBinding(error) => Some(error),
            Self::PhysicalLut { source, .. } => Some(source),
            Self::Stamp { source, .. } => Some(source),
            Self::MissingCandidateSet { .. }
            | Self::MissingCandidateColumn { .. }
            | Self::IncompatiblePorts { .. }
            | Self::SelectionCountMismatch { .. }
            | Self::CandidateIndexOutOfBounds { .. } => None,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::{Cursor, Write};

    use ndarray::{Array1, Array5, ArrayD, IxDyn};
    use ndarray_npy::WriteNpyExt;
    use shapeic_layout::PhysicalLookupTable;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::netlist::names::small_signal_param_name;

    use super::{
        CandidatePhysicalBinder, CandidatePhysicalBindingError, PhysicalCandidateStampOutcome,
    };
    use crate::macro_model::{ResolvedPhysicalPort, ResolvedPhysicalPrimitive};

    fn plan(ports: &[(&str, &str)]) -> ResolvedPhysicalPrimitive {
        ResolvedPhysicalPrimitive::new(
            "xcore",
            "core",
            "pair",
            "m1",
            ports
                .iter()
                .map(|(port, node)| ResolvedPhysicalPort::new(*port, *node))
                .collect(),
        )
    }

    fn candidates(lengths: &[f64]) -> CandidateSet {
        CandidateSet::new(
            "xcore",
            lengths
                .iter()
                .map(|length| {
                    CandidatePoint::new(
                        [
                            ("length", *length),
                            ("finger_width", 1.5),
                            ("nf", 1.0),
                            ("vbs", -0.5),
                            ("vgs", 1.0),
                            ("vds", 2.0),
                        ]
                        .map(|(parameter, value)| {
                            (small_signal_param_name(parameter, "xcore", "m1"), value)
                        })
                        .into(),
                    )
                })
                .collect(),
        )
    }

    pub(crate) fn physical_lut_for(primitive_name: &str, ports: &[&str]) -> PhysicalLookupTable {
        let cursor = Cursor::new(Vec::new());
        let mut archive = ZipWriter::new(cursor);
        let options = SimpleFileOptions::default();
        archive.start_file("manifest.json", options).unwrap();
        let port_order = ports
            .iter()
            .map(|port| format!("\"{port}\""))
            .collect::<Vec<_>>()
            .join(",");
        let manifest = format!(
            r#"{{
                    "format":"shapeic-physical-lut",
                    "version":2,
                    "pdk":"test",
                    "layout_policy":"test",
                    "primitives":[{{
                        "name":"{primitive_name}",
                        "port_order":[{port_order}],
                        "axis_order":["length","finger_width","nf"],
                        "axes":[
                            {{"name":"length","path":"p/l.npy"}},
                            {{"name":"finger_width","path":"p/w.npy"}},
                            {{"name":"nf","path":"p/n.npy"}}
                        ],
                        "conductance":"p/g.npy",
                        "capacitance":"p/c.npy",
                        "device_capacitance_correction":{{
                            "definition":"pex_mos_only_minus_aggregate_compact_model",
                            "axis_order":["length","finger_width","nf","vbs","vgs","vds"],
                            "axes":[
                                {{"name":"length","path":"p/d/l.npy"}},
                                {{"name":"finger_width","path":"p/d/w.npy"}},
                                {{"name":"nf","path":"p/d/n.npy"}},
                                {{"name":"vbs","path":"p/d/vbs.npy"}},
                                {{"name":"vgs","path":"p/d/vgs.npy"}},
                                {{"name":"vds","path":"p/d/vds.npy"}}
                            ],
                            "capacitance":"p/d/c.npy"
                        }}
                    }}]
                }}"#
        );
        archive.write_all(manifest.as_bytes()).unwrap();
        for (path, values) in [
            ("p/l.npy", vec![1.0, 2.0]),
            ("p/w.npy", vec![1.0, 2.0]),
            ("p/n.npy", vec![1.0, 2.0]),
            ("p/d/l.npy", vec![1.0, 2.0]),
            ("p/d/w.npy", vec![1.0, 2.0]),
            ("p/d/n.npy", vec![1.0, 2.0]),
            ("p/d/vbs.npy", vec![-1.0, 0.0]),
            ("p/d/vgs.npy", vec![0.0, 2.0]),
            ("p/d/vds.npy", vec![0.0, 4.0]),
        ] {
            let mut npy = Vec::new();
            Array1::from(values).write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let port_count = ports.len();
        let mut matrix = Array5::<f64>::zeros((2, 2, 2, port_count, port_count));
        for length_index in 0..2 {
            for width_index in 0..2 {
                for nf_index in 0..2 {
                    for row in 0..port_count {
                        for column in 0..port_count {
                            matrix[(length_index, width_index, nf_index, row, column)] =
                                if row == column {
                                    (port_count - 1) as f64
                                } else {
                                    -1.0
                                };
                        }
                    }
                }
            }
        }
        for path in ["p/g.npy", "p/c.npy"] {
            let mut npy = Vec::new();
            matrix.write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let correction = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2, 2, 2, 2, port_count, port_count]));
        let mut npy = Vec::new();
        correction.write_npy(&mut npy).unwrap();
        archive.start_file("p/d/c.npy", options).unwrap();
        archive.write_all(&npy).unwrap();

        PhysicalLookupTable::from_reader(Cursor::new(archive.finish().unwrap().into_inner()))
            .unwrap()
    }

    #[test]
    fn caches_queries_by_local_candidate_and_rejects_out_of_domain_points() {
        let lut = physical_lut_for("pair", &["P", "N"]);
        let candidates = candidates(&[1.5, 3.0]);
        let mut binder = CandidatePhysicalBinder::new(
            &[plan(&[("P", "VOUT"), ("N", "VSS")])],
            &[&candidates],
            &lut,
        )
        .unwrap();

        assert_eq!(
            binder.prepare(&[0]).unwrap(),
            PhysicalCandidateStampOutcome::Stamped
        );
        assert_eq!(
            binder.prepare(&[0]).unwrap(),
            PhysicalCandidateStampOutcome::Stamped
        );
        assert_eq!(binder.cached_query_count(), 1);
        assert_eq!(
            binder.prepare(&[1]).unwrap(),
            PhysicalCandidateStampOutcome::OutOfDomain
        );
        assert_eq!(binder.cached_query_count(), 2);
    }

    #[test]
    fn rejects_descriptor_ports_that_do_not_match_the_physical_lut() {
        let lut = physical_lut_for("pair", &["P", "N"]);
        let candidates = candidates(&[1.5]);
        let error = CandidatePhysicalBinder::new(
            &[plan(&[("P", "VOUT"), ("B", "VSS")])],
            &[&candidates],
            &lut,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CandidatePhysicalBindingError::IncompatiblePorts { .. }
        ));
    }
}
