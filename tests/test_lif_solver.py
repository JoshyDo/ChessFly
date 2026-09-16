"""Unit tests for the Rust-backed biophysical Leaky Integrate-and-Fire (LIF) solver."""

import time
import numpy as np
import pytest

from chessfly.simulator import LifSimulator


def test_lif_analytical_integration_and_threshold():
    """Verify subthreshold exponential decay and threshold crossing."""
    # 1 neuron with zero outgoing edges
    indptr = np.array([0, 0], dtype=np.uint64)
    indices = np.array([], dtype=np.uint32)
    weights = np.array([], dtype=np.float32)

    dt_ms = 0.2
    sim = LifSimulator(indptr, indices, weights, num_neurons=1, dt_ms=dt_ms)

    # Initial state should be at resting potential (-52 mV)
    v_init = sim.get_all_voltages()[0]
    assert np.isclose(v_init, -52.0, atol=1e-3)

    # Apply subthreshold current: 3.0 mV drive (target steady state = -49 mV, below -45 mV threshold)
    sim.set_sensory_drive(np.array([0], dtype=np.uint32), np.array([3.0], dtype=np.float32))
    sim.step(num_steps=500, dt_ms=dt_ms)  # 100 ms (5 * tau_m)

    v_sub = sim.get_all_voltages()[0]
    # Should settle near -52.0 + 3.0 = -49.0 mV
    assert np.isclose(v_sub, -49.0, atol=0.2)
    assert sim.get_all_spikes()[0] == 0, "No spikes should occur below threshold"

    # Now apply suprathreshold current: 15.0 mV drive (steady state = -37 mV > -45 mV)
    sim.set_sensory_drive(np.array([0], dtype=np.uint32), np.array([15.0], dtype=np.float32))
    sim.step(num_steps=500, dt_ms=dt_ms)  # 100 ms

    spikes = sim.get_all_spikes()[0]
    assert spikes > 0, "Neuron must fire action potentials when driven above threshold"


def test_motor_feature_formula():
    """Verify that Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV."""
    indptr = np.array([0, 0], dtype=np.uint64)
    indices = np.array([], dtype=np.uint32)
    weights = np.array([], dtype=np.float32)

    dt_ms = 0.2
    sim = LifSimulator(indptr, indices, weights, num_neurons=1, dt_ms=dt_ms)

    # Subthreshold current of 3.5 mV -> delta Vm = 3.5 mV
    # Feature should be: 0 spikes + 3.5 / 7.0 = 0.50
    sim.set_sensory_drive(np.array([0], dtype=np.uint32), np.array([3.5], dtype=np.float32))
    sim.step(num_steps=1000, dt_ms=dt_ms)  # 200 ms to reach steady state

    features = sim.get_motor_features(np.array([0], dtype=np.uint32))
    v = sim.get_all_voltages()[0]
    spikes = sim.get_all_spikes()[0]

    expected = spikes + (v - (-52.0)) / 7.0
    assert np.isclose(features[0], expected, atol=1e-3)
    assert 0.45 <= features[0] <= 0.55


def test_1000ms_simulation_window_performance():
    """Verify that a 1,000 ms simulation window executes rapidly."""
    # Graph with 1,000 neurons and 20,000 synapses
    n = 1000
    m = 20000
    rng = np.random.default_rng(42)

    pre = rng.integers(0, n, size=m, dtype=np.uint32)
    post = rng.integers(0, n, size=m, dtype=np.uint32)
    weights = rng.normal(5.0, 1.0, size=m).astype(np.float32)

    order = np.lexsort((post, pre))
    pre_s = pre[order]
    post_s = post[order]
    w_s = weights[order]

    counts = np.bincount(pre_s, minlength=n)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.uint64)

    sim = LifSimulator(indptr, post_s, w_s, num_neurons=n, dt_ms=0.5)

    t0 = time.perf_counter()
    # 1,000 ms at dt=0.5ms = 2,000 steps
    sim.simulate_window(duration_ms=1000.0, dt_ms=0.5)
    elapsed = time.perf_counter() - t0

    print(f"\n1,000 ms simulation of 1,000 neurons took: {elapsed*1000:.1f} ms")
    assert elapsed < 1.0, f"1,000 ms simulation must take under 1 second, took {elapsed:.2f}s"
