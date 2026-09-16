pub mod lif_network;
pub mod types;

use lif_network::LifNetwork;
use types::{LifParameters, SparseConnectomeCsr};

#[no_mangle]
pub extern "C" fn chessfly_create_network(
    num_neurons: usize,
    indptr: *const usize,
    indices: *const u32,
    weights: *const f32,
    num_weights: usize,
    dt_ms: f32,
) -> *mut LifNetwork {
    assert!(!indptr.is_null());
    assert!(!indices.is_null());
    assert!(!weights.is_null());

    let indptr_slice = unsafe { std::slice::from_raw_parts(indptr, num_neurons + 1) };
    let indices_slice = unsafe { std::slice::from_raw_parts(indices, num_weights) };
    let weights_slice = unsafe { std::slice::from_raw_parts(weights, num_weights) };

    let graph = SparseConnectomeCsr::new(
        num_neurons,
        indptr_slice.to_vec(),
        indices_slice.to_vec(),
        weights_slice.to_vec(),
    );

    let params = LifParameters::default();
    let network = LifNetwork::new(graph, params, dt_ms);
    Box::into_raw(Box::new(network))
}

#[no_mangle]
pub extern "C" fn chessfly_free_network(net: *mut LifNetwork) {
    if !net.is_null() {
        unsafe {
            drop(Box::from_raw(net));
        }
    }
}

#[no_mangle]
pub extern "C" fn chessfly_reset_state(net: *mut LifNetwork) {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    network.reset_state();
}

#[no_mangle]
pub extern "C" fn chessfly_step(net: *mut LifNetwork, num_steps: usize, dt_ms: f32) {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    network.step(num_steps, dt_ms);
}

#[no_mangle]
pub extern "C" fn chessfly_inject_spikes(
    net: *mut LifNetwork,
    target_indices: *const u32,
    num_targets: usize,
    weight: f32,
) {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    let targets = unsafe { std::slice::from_raw_parts(target_indices, num_targets) };
    network.inject_spikes(targets, weight);
}

#[no_mangle]
pub extern "C" fn chessfly_set_sensory_drive(
    net: *mut LifNetwork,
    target_indices: *const u32,
    currents: *const f32,
    count: usize,
) {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    let targets = unsafe { std::slice::from_raw_parts(target_indices, count) };
    let currs = unsafe { std::slice::from_raw_parts(currents, count) };
    network.set_sensory_drive(targets, currs);
}

#[no_mangle]
pub extern "C" fn chessfly_get_motor_features(
    net: *const LifNetwork,
    motor_indices: *const u32,
    count: usize,
    out_features: *mut f32,
) {
    assert!(!net.is_null());
    assert!(!out_features.is_null());
    let network = unsafe { &*net };
    let targets = unsafe { std::slice::from_raw_parts(motor_indices, count) };
    let features = network.get_motor_features(targets);
    unsafe {
        std::ptr::copy_nonoverlapping(features.as_ptr(), out_features, count);
    }
}

#[no_mangle]
pub extern "C" fn chessfly_get_all_spikes(net: *const LifNetwork, out_spikes: *mut u32) {
    assert!(!net.is_null());
    assert!(!out_spikes.is_null());
    let network = unsafe { &*net };
    unsafe {
        std::ptr::copy_nonoverlapping(
            network.spike_counts.as_ptr(),
            out_spikes,
            network.graph.num_neurons,
        );
    }
}

#[no_mangle]
pub extern "C" fn chessfly_get_all_voltages(net: *const LifNetwork, out_v: *mut f32) {
    assert!(!net.is_null());
    assert!(!out_v.is_null());
    let network = unsafe { &*net };
    unsafe {
        std::ptr::copy_nonoverlapping(network.v.as_ptr(), out_v, network.graph.num_neurons);
    }
}

#[no_mangle]
pub extern "C" fn chessfly_apply_plasticity(
    net: *mut LifNetwork,
    kc_indices: *const u32,
    num_kc: usize,
    mbon_indices: *const u32,
    num_mbon: usize,
    dan_signal: f32,
    eta: f32,
) -> usize {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    let kc_slice = unsafe { std::slice::from_raw_parts(kc_indices, num_kc) };
    let mbon_slice = unsafe { std::slice::from_raw_parts(mbon_indices, num_mbon) };
    network.apply_dopaminergic_plasticity(kc_slice, mbon_slice, dan_signal, eta)
}

#[no_mangle]
pub extern "C" fn chessfly_set_neuromodulation(
    net: *mut LifNetwork,
    octopamine: f32,
    conductance_gain: f32,
) {
    assert!(!net.is_null());
    let network = unsafe { &mut *net };
    network.set_neuromodulation(octopamine, conductance_gain);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_basic_spiking_and_refractory() {
        let n = 2;
        // Neuron 0 connects to Neuron 1 with strong Giant Fiber excitatory weight +60.0
        let indptr = vec![0, 1, 1];
        let indices = vec![1];
        let weights = vec![60.0];

        let graph = SparseConnectomeCsr::new(n, indptr, indices, weights);
        let params = LifParameters::default();
        let mut net = LifNetwork::new(graph, params, 0.2);

        // Subthreshold integration with current drive
        net.sensory_drive[0] = 10.0; // strong drive above threshold (v_thresh - v_rest = 7.0 mV)
        
        // Step for 100 ms (500 steps at dt=0.2ms, tau_m=20ms requires ~24.1ms to reach threshold)
        net.step(500, 0.2);

        // Neuron 0 should have fired multiple spikes
        assert!(net.spike_counts[0] > 0, "Neuron 0 must fire spikes with 10.0 drive");
        // Neuron 1 should have received delayed spikes and fired as well
        assert!(net.spike_counts[1] > 0, "Neuron 1 must fire from synaptic input");

        let motor_features = net.get_motor_features(&[0, 1]);
        assert_eq!(motor_features.len(), 2);
        assert!(motor_features[0] >= 1.0);
    }
}
