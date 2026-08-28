use std::error::Error;
use std::fmt;

const DEFAULT_BATCHES_PER_WORKER: usize = 8;

/// Runtime policy used to execute one macro exploration.
///
/// Exploration remains sequential unless parallel electrical analysis is
/// enabled explicitly. Candidate construction has a separate future control
/// and is intentionally unaffected by this configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MacroExecutionConfig {
    electrical_analysis: ElectricalAnalysisExecution,
}

impl MacroExecutionConfig {
    /// Creates the default sequential execution policy.
    pub const fn sequential() -> Self {
        Self {
            electrical_analysis: ElectricalAnalysisExecution::Sequential,
        }
    }

    /// Enables batched electrical analysis using a local pool of `workers`.
    ///
    /// The initial batch size is eight batches per worker and can be replaced
    /// with [`Self::with_electrical_batch_size`]. A single worker deliberately
    /// selects the sequential baseline.
    pub fn with_parallel_electrical_analysis(
        mut self,
        workers: usize,
    ) -> Result<Self, MacroExecutionConfigError> {
        if workers == 0 {
            return Err(MacroExecutionConfigError::ZeroElectricalWorkers);
        }
        if workers == 1 {
            self.electrical_analysis = ElectricalAnalysisExecution::Sequential;
            return Ok(self);
        }
        let batch_size = workers
            .checked_mul(DEFAULT_BATCHES_PER_WORKER)
            .ok_or(MacroExecutionConfigError::ElectricalBatchSizeOverflow { workers })?;
        self.electrical_analysis = ElectricalAnalysisExecution::Parallel {
            workers,
            batch_size,
        };
        Ok(self)
    }

    /// Replaces the batch size of enabled parallel electrical analysis.
    pub fn with_electrical_batch_size(
        mut self,
        batch_size: usize,
    ) -> Result<Self, MacroExecutionConfigError> {
        if batch_size == 0 {
            return Err(MacroExecutionConfigError::ZeroElectricalBatchSize);
        }
        let ElectricalAnalysisExecution::Parallel { workers, .. } = self.electrical_analysis else {
            return Err(MacroExecutionConfigError::ParallelElectricalAnalysisNotEnabled);
        };
        self.electrical_analysis = ElectricalAnalysisExecution::Parallel {
            workers,
            batch_size,
        };
        Ok(self)
    }

    /// Returns the configured electrical-analysis execution policy.
    pub const fn electrical_analysis(&self) -> ElectricalAnalysisExecution {
        self.electrical_analysis
    }
}

/// Execution mode for the electrical prefix of a macro's testbenches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ElectricalAnalysisExecution {
    #[default]
    Sequential,
    Parallel {
        workers: usize,
        batch_size: usize,
    },
}

/// Invalid macro execution configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroExecutionConfigError {
    ZeroElectricalWorkers,
    ZeroElectricalBatchSize,
    ElectricalBatchSizeOverflow { workers: usize },
    ParallelElectricalAnalysisNotEnabled,
}

impl fmt::Display for MacroExecutionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroElectricalWorkers => {
                formatter.write_str("electrical analysis worker count must be greater than zero")
            }
            Self::ZeroElectricalBatchSize => {
                formatter.write_str("electrical analysis batch size must be greater than zero")
            }
            Self::ElectricalBatchSizeOverflow { workers } => write!(
                formatter,
                "default electrical analysis batch size overflows for {workers} workers"
            ),
            Self::ParallelElectricalAnalysisNotEnabled => formatter.write_str(
                "electrical batch size requires parallel electrical analysis to be enabled",
            ),
        }
    }
}

impl Error for MacroExecutionConfigError {}

#[cfg(test)]
mod tests {
    use super::{ElectricalAnalysisExecution, MacroExecutionConfig, MacroExecutionConfigError};

    #[test]
    fn defaults_to_sequential_execution() {
        assert_eq!(
            MacroExecutionConfig::default().electrical_analysis(),
            ElectricalAnalysisExecution::Sequential
        );
    }

    #[test]
    fn configures_parallel_electrical_workers_and_batches() {
        let execution = MacroExecutionConfig::sequential()
            .with_parallel_electrical_analysis(4)
            .unwrap();
        assert_eq!(
            execution.electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 4,
                batch_size: 32,
            }
        );

        let execution = execution.with_electrical_batch_size(7).unwrap();
        assert_eq!(
            execution.electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 4,
                batch_size: 7,
            }
        );
    }

    #[test]
    fn one_worker_selects_the_sequential_baseline() {
        let execution = MacroExecutionConfig::sequential()
            .with_parallel_electrical_analysis(1)
            .unwrap();
        assert_eq!(
            execution.electrical_analysis(),
            ElectricalAnalysisExecution::Sequential
        );
    }

    #[test]
    fn rejects_invalid_parallel_settings() {
        assert_eq!(
            MacroExecutionConfig::sequential().with_parallel_electrical_analysis(0),
            Err(MacroExecutionConfigError::ZeroElectricalWorkers)
        );
        assert_eq!(
            MacroExecutionConfig::sequential().with_electrical_batch_size(4),
            Err(MacroExecutionConfigError::ParallelElectricalAnalysisNotEnabled)
        );
        assert_eq!(
            MacroExecutionConfig::sequential()
                .with_parallel_electrical_analysis(2)
                .unwrap()
                .with_electrical_batch_size(0),
            Err(MacroExecutionConfigError::ZeroElectricalBatchSize)
        );
    }
}
