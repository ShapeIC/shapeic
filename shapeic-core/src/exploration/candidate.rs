#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSet {
    pub name: String,
    pub points: Vec<CandidatePoint>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePoint {
    pub values: Vec<(String, f64)>,
}
