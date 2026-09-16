/// High-performance biophysical Leaky Integrate-and-Fire (LIF) network solver.

use crate::types::{LifParameters, SparseConnectomeCsr};

pub struct LifNetwork {
    pub params: LifParameters,
    pub graph: SparseConnectomeCsr,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub refractory_steps_remaining: Vec<u16>,
    pub spike_counts: Vec<u32>,
    pub sensory_drive: Vec<f32>,
    pub v_integrated: Vec<f64>,

    // Ring-buffer delay queue: delay_steps = round(transmission_delay / dt)
    // Ring buffer size = delay_steps + 1
    pub delay_queue: Vec<Vec<u32>>,
    pub queue_slot: usize,
    pub delay_steps: usize,
    pub refractory_steps: u16,

    // Step counter
    pub total_steps: u64,
}

impl LifNetwork {
    pub fn new(graph: SparseConnectomeCsr, params: LifParameters, dt_ms: f32) -> Self {
        let n = graph.num_neurons;
        let delay_steps = (params.transmission_delay_ms / dt_ms).round().max(1.0) as usize;
        let refractory_steps = (params.refractory_period_ms / dt_ms).round().max(1.0) as u16;
        let num_slots = delay_steps + 1;

        let delay_queue = (0..num_slots).map(|_| Vec::with_capacity(128)).collect();

        Self {
            v: vec![params.v_rest; n],
            g: vec![0.0; n],
            refractory_steps_remaining: vec![0; n],
            spike_counts: vec![0; n],
            sensory_drive: vec![0.0; n],
            v_integrated: vec![0.0; n],
            delay_queue,
            queue_slot: 0,
            delay_steps,
            refractory_steps,
            total_steps: 0,
            params,
            graph,
        }
    }

    /// Reset membrane potential and activity back to resting state.
    pub fn reset_state(&mut self) {
        self.v.fill(self.params.v_rest);
        self.g.fill(0.0);
        self.refractory_steps_remaining.fill(0);
        self.spike_counts.fill(0);
        self.sensory_drive.fill(0.0);
        self.v_integrated.fill(0.0);
        for slot in &mut self.delay_queue {
            slot.clear();
        }
        self.queue_slot = 0;
        self.total_steps = 0;
    }

    /// Set continuous sensory drive currents (e.g. for background or visual input).
    pub fn set_sensory_drive(&mut self, indices: &[u32], currents: &[f32]) {
        assert_eq!(indices.len(), currents.len());
        for (&idx, &curr) in indices.iter().zip(currents.iter()) {
            if (idx as usize) < self.sensory_drive.len() {
                self.sensory_drive[idx as usize] = curr;
            }
        }
    }

    /// Inject immediate sensory spikes into target neurons (e.g. from 180 Hz Poisson trains).
    pub fn inject_spikes(&mut self, target_indices: &[u32], weight: f32) {
        for &idx in target_indices {
            let i = idx as usize;
            if i < self.g.len() && self.refractory_steps_remaining[i] == 0 {
                self.g[i] += weight;
            }
        }
    }

    /// Advance the LIF network by `num_steps` of size `dt_ms`.
    pub fn step(&mut self, num_steps: usize, dt_ms: f32) {
        let a = (-dt_ms / self.params.tau_m).exp();
        let b = (-dt_ms / self.params.tau_s).exp();
        // Exact integral factor for g(t) = g0 * exp(-t / tau_s) into dv/dt = -(v - v_rest) / tau_m + g:
        // tau_s / (tau_m - tau_s) = 5.0 / (20.0 - 5.0) = 1.0 / 3.0
        let g_coeff = (a - b) / 3.0;
        let drive_coeff = 1.0 - a;
        let v_rest = self.params.v_rest;
        let v_thresh = self.params.v_thresh;
        let n = self.graph.num_neurons;
        let num_slots = self.delay_queue.len();

        for _ in 0..num_steps {
            let current_slot = self.queue_slot;
            let future_slot = (self.queue_slot + self.delay_steps) % num_slots;

            // 1. Deliver delayed spikes that arrived at the current time slot
            // Drain the current slot's queued spikes
            let queued_spikes = std::mem::take(&mut self.delay_queue[current_slot]);
            for pre_neuron in queued_spikes {
                let (post_indices, post_weights) = self.graph.outgoing_edges(pre_neuron as usize);
                for (&post_idx, &weight) in post_indices.iter().zip(post_weights.iter()) {
                    let j = post_idx as usize;
                    if j < n && self.refractory_steps_remaining[j] == 0 {
                        self.g[j] += weight;
                    }
                }
            }

            // 2. Update membrane potential and conductance for all neurons
            for i in 0..n {
                // Decay synaptic conductance
                let prev_g = self.g[i];
                self.g[i] = prev_g * b;

                // If in refractory period, hold at resting potential and decrement counter
                if self.refractory_steps_remaining[i] > 0 {
                    self.refractory_steps_remaining[i] -= 1;
                    self.v[i] = v_rest;
                    continue;
                }

                // Subthreshold integration
                let v_prev = self.v[i];
                let drive = self.sensory_drive[i];
                let v_next = v_rest + (v_prev - v_rest) * a + drive * drive_coeff + prev_g * g_coeff;

                // Track accumulated voltage shift for readout
                self.v_integrated[i] += (v_next - v_rest) as f64;

                // Check for action potential
                if v_next >= v_thresh {
                    // Spike fired!
                    self.spike_counts[i] += 1;
                    self.v[i] = v_rest;
                    self.refractory_steps_remaining[i] = self.refractory_steps;

                    // Schedule spike delivery in the future slot
                    self.delay_queue[future_slot].push(i as u32);
                } else {
                    self.v[i] = v_next;
                }
            }

            self.queue_slot = (self.queue_slot + 1) % num_slots;
            self.total_steps += 1;
        }
    }

    /// Read motor readout features: Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV
    pub fn get_motor_features(&self, motor_indices: &[u32]) -> Vec<f32> {
        let v_rest = self.params.v_rest;
        motor_indices
            .iter()
            .map(|&idx| {
                let i = idx as usize;
                if i < self.graph.num_neurons {
                    let spikes = self.spike_counts[i] as f32;
                    let delta_v = (self.v[i] - v_rest).clamp(0.0, 7.0);
                    spikes + delta_v / 7.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Dopaminergic plasticity update across KC -> MBON synapses.
    /// dan_signal: +1.0 for PAM11 reward, -1.0 for PPL101 aversive
    pub fn apply_dopaminergic_plasticity(
        &mut self,
        kc_indices: &[u32],
        mbon_indices: &[u32],
        dan_signal: f32,
        eta: f32,
    ) -> usize {
        let mut modified_count = 0;
        let is_mbon = |target: u32| mbon_indices.contains(&target);

        for &kc in kc_indices {
            let pre = kc as usize;
            if pre >= self.graph.num_neurons {
                continue;
            }
            let pre_activity = self.spike_counts[pre] as f32;
            if pre_activity <= 0.0 {
                continue;
            }

            let start = self.graph.indptr[pre];
            let end = self.graph.indptr[pre + 1];

            for edge_idx in start..end {
                let post = self.graph.indices[edge_idx];
                if is_mbon(post) {
                    let post_activity = self.spike_counts[post as usize] as f32;
                    // Delta W = eta * dan_signal * r_pre * (1.0 + r_post)
                    let delta_w = eta * dan_signal * pre_activity * (1.0 + 0.1 * post_activity);
                    // Update synaptic weight with bounding
                    let new_w = (self.graph.weights[edge_idx] + delta_w).clamp(0.01, 10.0);
                    self.graph.weights[edge_idx] = new_w;
                    modified_count += 1;
                }
            }
        }
        modified_count
    }
}
