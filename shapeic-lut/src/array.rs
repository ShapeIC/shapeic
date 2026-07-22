use ndarray::{ArrayD, ArrayViewD, IxDyn};

/// Numeric element type retained from the source NumPy array.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DType {
    F32,
    F64,
}

/// An owned, dynamically shaped numeric array from a LUT.
#[derive(Clone, Debug, PartialEq)]
pub enum LutArray {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

impl LutArray {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(array) => array.shape(),
            Self::F64(array) => array.shape(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(array) => array.len(),
            Self::F64(array) => array.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Option<ArrayViewD<'_, f32>> {
        match self {
            Self::F32(array) => Some(array.view()),
            Self::F64(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<ArrayViewD<'_, f64>> {
        match self {
            Self::F32(_) => None,
            Self::F64(array) => Some(array.view()),
        }
    }

    pub fn get_f64(&self, index: &[usize]) -> Option<f64> {
        match self {
            Self::F32(array) => array.get(IxDyn(index)).map(|value| f64::from(*value)),
            Self::F64(array) => array.get(IxDyn(index)).copied(),
        }
    }

    pub(crate) fn scalar_or_grid_value(&self, index: &[usize]) -> Option<f64> {
        if self.shape() == [1] {
            self.get_f64(&[0])
        } else {
            self.get_f64(index)
        }
    }

    pub(crate) fn into_f64_vec(self) -> Vec<f64> {
        match self {
            Self::F32(array) => array.into_iter().map(f64::from).collect(),
            Self::F64(array) => array.into_iter().collect(),
        }
    }
}
