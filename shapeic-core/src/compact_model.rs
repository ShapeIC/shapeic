//! Compact-model data and numerical MNA stamping utilities.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use ndarray::Array2;
use shapeic_lut::{LutError, MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_mna::numeric::{NumericMnaError, NumericMnaSystem};

use crate::macro_model::ResolvedPrimitiveBranch;

/// Complete intrinsic and extrinsic capacitance data for one sized MOS device.
///
/// Both components represent the complete device. The intrinsic matrix must
/// already include finger-count scaling, and the extrinsic values must be the
/// total values returned by current-based sizing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosDeviceCapacitances {
    /// Total intrinsic nodal capacitance matrix in terminal order G, D, S, B.
    pub intrinsic: MosCapacitanceMatrix,
    /// Total overlap and junction capacitances.
    pub extrinsic: MosExtrinsicCapacitances,
}

impl MosDeviceCapacitances {
    /// Constructs a device from nine total intrinsic ngspice coefficients and
    /// four total extrinsic capacitances.
    pub fn from_total_parameters(
        intrinsic: &[f64],
        extrinsic: MosExtrinsicCapacitances,
    ) -> Result<Self, LutError> {
        let extrinsic_values = [extrinsic.cgsol, extrinsic.cgdol, extrinsic.cjs, extrinsic.cjd];
        if extrinsic_values.iter().any(|value| !value.is_finite()) {
            return Err(LutError::InvalidExtrinsicCapacitances {
                reason: "all total extrinsic capacitances must be finite".to_owned(),
            });
        }
        Ok(Self {
            intrinsic: MosCapacitanceMatrix::from_independent_ngspice_parameters(intrinsic)?,
            extrinsic,
        })
    }

    /// Returns the complete 4×4 nodal capacitance matrix in terminal order
    /// G, D, S, B.
    pub fn combined_matrix(self) -> [[f64; 4]; 4] {
        let mut matrix = self.intrinsic.values;
        add_passive_capacitance(&mut matrix, 0, 2, self.extrinsic.cgsol);
        add_passive_capacitance(&mut matrix, 0, 1, self.extrinsic.cgdol);
        add_passive_capacitance(&mut matrix, 2, 3, self.extrinsic.cjs);
        add_passive_capacitance(&mut matrix, 1, 3, self.extrinsic.cjd);
        matrix
    }

    /// Stamps the complete MOS capacitance matrix into an instantiated
    /// numerical MNA system.
    pub fn stamp(
        self,
        system: &mut NumericMnaSystem,
        connections: [(&str, &str); 4],
    ) -> Result<(), NumericMnaError> {
        let matrix = self.combined_matrix();
        let capacitance = Array2::from_shape_fn((4, 4), |(row, column)| matrix[row][column]);
        let conductance = Array2::<f64>::zeros((4, 4));
        let ports = MosCapacitanceMatrix::TERMINALS.map(str::to_owned).to_vec();
        let connections = connections
            .into_iter()
            .map(|(port, node)| (port.to_owned(), node.to_owned()))
            .collect::<BTreeMap<_, _>>();
        system.stamp_port_admittance(
            &ports,
            &connections,
            conductance.view(),
            capacitance.view(),
        )
    }
}

/// Stamps complete MOS capacitances using resolved hierarchical branch nodes.
///
/// `capacitances` must follow `branches` exactly. The count is checked before
/// the MNA system is modified.
pub fn stamp_resolved_mos_capacitances(
    system: &mut NumericMnaSystem,
    branches: &[ResolvedPrimitiveBranch],
    capacitances: &[MosDeviceCapacitances],
) -> Result<(), ResolvedCapacitanceStampError> {
    if branches.len() != capacitances.len() {
        return Err(ResolvedCapacitanceStampError::CountMismatch {
            branches: branches.len(),
            capacitances: capacitances.len(),
        });
    }

    for (branch, capacitances) in branches.iter().zip(capacitances) {
        capacitances
            .stamp(
                system,
                [
                    ("G", branch.gate_node()),
                    ("D", branch.drain_node()),
                    ("S", branch.source_node()),
                    ("B", branch.bulk_node()),
                ],
            )
            .map_err(|source| ResolvedCapacitanceStampError::Branch {
                instance_path: branch.instance_path().to_owned(),
                branch: branch.branch_name().to_owned(),
                source,
            })?;
    }
    Ok(())
}

/// Errors produced while stamping capacitances for resolved primitive branches.
#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedCapacitanceStampError {
    /// Branch topology and capacitance values have different lengths.
    CountMismatch {
        branches: usize,
        capacitances: usize,
    },
    /// A branch could not be stamped into the numerical MNA system.
    Branch {
        instance_path: String,
        branch: String,
        source: NumericMnaError,
    },
}

impl fmt::Display for ResolvedCapacitanceStampError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountMismatch {
                branches,
                capacitances,
            } => write!(
                formatter,
                "resolved topology contains {branches} branches but {capacitances} capacitance values were provided"
            ),
            Self::Branch {
                instance_path,
                branch,
                source,
            } => write!(
                formatter,
                "could not stamp capacitances for '{instance_path}.{branch}': {source}"
            ),
        }
    }
}

impl Error for ResolvedCapacitanceStampError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Branch { source, .. } => Some(source),
            Self::CountMismatch { .. } => None,
        }
    }
}

fn add_passive_capacitance(
    matrix: &mut [[f64; 4]; 4],
    positive: usize,
    negative: usize,
    capacitance: f64,
) {
    matrix[positive][positive] += capacitance;
    matrix[negative][negative] += capacitance;
    matrix[positive][negative] -= capacitance;
    matrix[negative][positive] -= capacitance;
}

#[cfg(test)]
mod tests {
    use shapeic_mna::mna::mna_from_spice;
    use shapeic_mna::numeric::PreparedNumericMna;

    use super::*;

    const INTRINSIC: [f64; 9] = [
        10.0e-15, 2.0e-15, 3.0e-15, 1.0e-15, 8.0e-15, 2.0e-15, 1.5e-15, 2.5e-15,
        7.0e-15,
    ];
    const EXTRINSIC: MosExtrinsicCapacitances = MosExtrinsicCapacitances {
        cgsol: 1.0e-15,
        cgdol: 2.0e-15,
        cjs: 3.0e-15,
        cjd: 4.0e-15,
    };

    fn capacitances() -> MosDeviceCapacitances {
        MosDeviceCapacitances::from_total_parameters(&INTRINSIC, EXTRINSIC)
            .expect("valid total capacitances")
    }

    fn branch(instance: &str, branch: &str, nodes: [&str; 4]) -> ResolvedPrimitiveBranch {
        ResolvedPrimitiveBranch::new(
            instance.to_owned(),
            "mos_primitive".to_owned(),
            branch.to_owned(),
            nodes[0].to_owned(),
            nodes[1].to_owned(),
            nodes[2].to_owned(),
            nodes[3].to_owned(),
        )
    }

    #[test]
    fn combines_intrinsic_overlap_and_junction_capacitances() {
        let capacitances = capacitances();
        let intrinsic = capacitances.intrinsic.values;
        let combined = capacitances.combined_matrix();

        assert_eq!(combined[0][0], intrinsic[0][0] + EXTRINSIC.cgsol + EXTRINSIC.cgdol);
        assert_eq!(combined[0][1], intrinsic[0][1] - EXTRINSIC.cgdol);
        assert_eq!(combined[0][2], intrinsic[0][2] - EXTRINSIC.cgsol);
        assert_eq!(combined[1][1], intrinsic[1][1] + EXTRINSIC.cgdol + EXTRINSIC.cjd);
        assert_eq!(combined[2][2], intrinsic[2][2] + EXTRINSIC.cgsol + EXTRINSIC.cjs);
        assert_eq!(combined[3][3], intrinsic[3][3] + EXTRINSIC.cjs + EXTRINSIC.cjd);
        for index in 0..4 {
            assert!(combined[index].iter().sum::<f64>().abs() < 1.0e-27);
            assert!(combined.iter().map(|row| row[index]).sum::<f64>().abs() < 1.0e-27);
        }
    }

    #[test]
    fn rejects_non_finite_total_parameters() {
        let mut intrinsic = INTRINSIC;
        intrinsic[0] = f64::NAN;
        assert!(matches!(
            MosDeviceCapacitances::from_total_parameters(&intrinsic, EXTRINSIC),
            Err(LutError::InvalidCapacitanceMatrix { .. })
        ));

        let mut extrinsic = EXTRINSIC;
        extrinsic.cjd = f64::INFINITY;
        assert!(matches!(
            MosDeviceCapacitances::from_total_parameters(&INTRINSIC, extrinsic),
            Err(LutError::InvalidExtrinsicCapacitances { .. })
        ));
    }

    #[test]
    fn stamps_only_the_capacitance_part_of_the_numeric_mna() {
        let mna = mna_from_spice(
            "Vg G_NODE VSS 0\nVd D_NODE VSS 0\nVs S_NODE VSS 0\nVb B_NODE VSS 0\n\
             Vg2 G2 VSS 0\nVd2 D2 VSS 0\nVs2 S2 VSS 0\nVb2 B2 VSS 0\n.end",
        )
        .expect("four-terminal test circuit should parse");
        let node_indices = ["G_NODE", "D_NODE", "S_NODE", "B_NODE"].map(|node| {
            mna.nodes
                .get(node)
                .expect("test node should exist")
                .checked_sub(1)
                .expect("test terminal must not be ground")
        });
        let mut prepared = PreparedNumericMna::new(&mna, &[]).expect("MNA should prepare");
        let mut system = prepared.instantiate(&[]).expect("MNA should instantiate");
        let base_before = system.base_matrix().clone();
        let expected = capacitances().combined_matrix();

        capacitances()
            .stamp(
                &mut system,
                [
                    ("G", "G_NODE"),
                    ("D", "D_NODE"),
                    ("S", "S_NODE"),
                    ("B", "B_NODE"),
                ],
            )
            .expect("MOS capacitance stamp should succeed");

        assert_eq!(system.base_matrix(), &base_before);
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(
                    system.capacitance_matrix()[(node_indices[row], node_indices[column])],
                    expected[row][column]
                );
            }
        }

        let mut resolved_system = prepared.instantiate(&[]).unwrap();
        let mut expected_system = prepared.instantiate(&[]).unwrap();
        let branches = [
            branch("xdp", "m1", ["G_NODE", "D_NODE", "S_NODE", "B_NODE"]),
            branch("xcm", "m2", ["G2", "D2", "S2", "B2"]),
        ];
        let values = [capacitances(), capacitances()];

        stamp_resolved_mos_capacitances(&mut resolved_system, &branches, &values).unwrap();
        values[0]
            .stamp(
                &mut expected_system,
                [
                    ("G", "G_NODE"),
                    ("D", "D_NODE"),
                    ("S", "S_NODE"),
                    ("B", "B_NODE"),
                ],
            )
            .unwrap();
        values[1]
            .stamp(
                &mut expected_system,
                [("G", "G2"), ("D", "D2"), ("S", "S2"), ("B", "B2")],
            )
            .unwrap();

        assert_eq!(
            resolved_system.capacitance_matrix(),
            expected_system.capacitance_matrix()
        );

        assert_eq!(
            stamp_resolved_mos_capacitances(
                &mut resolved_system,
                std::slice::from_ref(&branches[0]),
                &[]
            ),
            Err(ResolvedCapacitanceStampError::CountMismatch {
                branches: 1,
                capacitances: 0,
            })
        );
        let missing_bulk = branch(
            "xmissing",
            "m3",
            ["G_NODE", "D_NODE", "S_NODE", "MISSING_BULK"],
        );
        assert!(matches!(
            stamp_resolved_mos_capacitances(
                &mut resolved_system,
                &[missing_bulk],
                &[capacitances()]
            ),
            Err(ResolvedCapacitanceStampError::Branch {
                instance_path,
                branch,
                source: NumericMnaError::MissingNode(node),
            }) if instance_path == "xmissing" && branch == "m3" && node == "MISSING_BULK"
        ));
    }
}
