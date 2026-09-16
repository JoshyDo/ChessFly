/// Type definitions and configuration for the biophysical LIF solver.

#[derive(Debug, Clone)]
pub struct LifParameters {
    /// Resting membrane potential (mV)
    pub v_rest: f32,
    /// Action potential firing threshold (mV)
    pub v_thresh: f32,
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Synaptic conductance decay time constant (ms)
    pub tau_s: f32,
    /// Refractory period duration (ms)
    pub refractory_period_ms: f32,
    /// Axonal and synaptic transmission delay (ms)
    pub transmission_delay_ms: f32,
}

impl Default for LifParameters {
    fn default() -> Self {
        Self {
            v_rest: -52.0,
            v_thresh: -45.0,
            tau_m: 20.0,
            tau_s: 5.0,
            refractory_period_ms: 2.2,
            transmission_delay_ms: 1.8,
        }
    }
}

/// Sparse CSR connectivity graph storing directed synaptic connections.
#[derive(Debug, Clone)]
pub struct SparseConnectomeCsr {
    pub num_neurons: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

impl SparseConnectomeCsr {
    pub fn new(num_neurons: usize, indptr: Vec<usize>, indices: Vec<u32>, weights: Vec<f32>) -> Self {
        assert_eq!(indptr.len(), num_neurons + 1, "indptr must have length num_neurons + 1");
        assert_eq!(indices.len(), weights.len(), "indices and weights must have identical length");
        Self {
            num_neurons,
            indptr,
            indices,
            weights,
        }
    }

    #[inline(always)]
    pub fn outgoing_edges(&self, neuron: usize) -> (&[u32], &[f32]) {
        let start = self.indptr[neuron];
        let end = self.indptr[neuron + 1];
        (&self.indices[start..end], &self.weights[start..end])
    }
}
