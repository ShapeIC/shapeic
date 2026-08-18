//! Compact-model data and numerical MNA stamping utilities.

use std::collections::BTreeMap;

use ndarray::Array2;
use shapeic_lut::{LutError, MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_mna::numeric::{NumericMnaError, NumericMnaSystem};

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
            "Vg G_NODE VSS 0\nVd D_NODE VSS 0\nVs S_NODE VSS 0\nVb B_NODE VSS 0\n.end",
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
    }
}
