"""Python ctypes wrapper for the high-performance Rust LIF connectome solver."""

import ctypes as C
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def find_library() -> Path:
    """Find the compiled chessfly_core dynamic library."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "target" / "release" / "libchessfly_core.dylib",
        root / "target" / "release" / "libchessfly_core.so",
        root / "target" / "debug" / "libchessfly_core.dylib",
        root / "target" / "debug" / "libchessfly_core.so",
        Path(__file__).parent / "libchessfly_core.dylib",
        Path(__file__).parent / "libchessfly_core.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find libchessfly_core dynamic library. Looked in: {[str(c) for c in candidates]}. "
        "Run `cargo build --release` first."
    )


class LifSimulator:
    """High-speed Drosophila connectome LIF simulator backed by the Rust engine."""

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        num_neurons: Optional[int] = None,
        dt_ms: float = 0.2,
    ):
        self.lib_path = find_library()
        self._lib = C.CDLL(str(self.lib_path))
        self.dt_ms = float(dt_ms)

        # Setup ctypes prototypes
        self._lib.chessfly_create_network.argtypes = [
            C.c_size_t,
            C.POINTER(C.c_size_t),
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_float),
            C.c_size_t,
            C.c_float,
        ]
        self._lib.chessfly_create_network.restype = C.c_void_p

        self._lib.chessfly_free_network.argtypes = [C.c_void_p]
        self._lib.chessfly_free_network.restype = None

        self._lib.chessfly_reset_state.argtypes = [C.c_void_p]
        self._lib.chessfly_reset_state.restype = None

        self._lib.chessfly_step.argtypes = [C.c_void_p, C.c_size_t, C.c_float]
        self._lib.chessfly_step.restype = None

        self._lib.chessfly_inject_spikes.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_float,
        ]
        self._lib.chessfly_inject_spikes.restype = None

        self._lib.chessfly_set_sensory_drive.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_float),
            C.c_size_t,
        ]
        self._lib.chessfly_set_sensory_drive.restype = None

        self._lib.chessfly_get_motor_features.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.POINTER(C.c_float),
        ]
        self._lib.chessfly_get_motor_features.restype = None

        self._lib.chessfly_get_all_spikes.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
        ]
        self._lib.chessfly_get_all_spikes.restype = None

        self._lib.chessfly_get_all_voltages.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_float),
        ]
        self._lib.chessfly_get_all_voltages.restype = None

        self._lib.chessfly_apply_plasticity.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.POINTER(C.c_uint32),
            C.c_size_t,
            C.c_float,
            C.c_float,
        ]
        self._lib.chessfly_apply_plasticity.restype = C.c_size_t

        self._lib.chessfly_set_neuromodulation.argtypes = [
            C.c_void_p,
            C.c_float,
            C.c_float,
        ]
        self._lib.chessfly_set_neuromodulation.restype = None

        # Prepare numpy arrays
        self.indptr = np.ascontiguousarray(indptr, dtype=np.uint64)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32)
        self.weights = np.ascontiguousarray(weights, dtype=np.float32)

        if num_neurons is None:
            num_neurons = len(self.indptr) - 1
        self.num_neurons = int(num_neurons)

        ptr_p = self.indptr.ctypes.data_as(C.POINTER(C.c_size_t))
        idx_p = self.indices.ctypes.data_as(C.POINTER(C.c_uint32))
        w_p = self.weights.ctypes.data_as(C.POINTER(C.c_float))

        self._handle = self._lib.chessfly_create_network(
            self.num_neurons,
            ptr_p,
            idx_p,
            w_p,
            len(self.weights),
            C.c_float(self.dt_ms),
        )

        if not self._handle:
            raise RuntimeError("Failed to create native LifNetwork in Rust.")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.chessfly_free_network(self._handle)
            self._handle = None

    def reset(self):
        """Reset membrane potential, queues, and spike counts back to rest."""
        self._lib.chessfly_reset_state(self._handle)

    def set_sensory_drive(self, indices: np.ndarray, currents: np.ndarray):
        """Set continuous sensory currents on target neurons."""
        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        cur = np.ascontiguousarray(currents, dtype=np.float32)
        self._lib.chessfly_set_sensory_drive(
            self._handle,
            idx.ctypes.data_as(C.POINTER(C.c_uint32)),
            cur.ctypes.data_as(C.POINTER(C.c_float)),
            len(idx),
        )

    def inject_spikes(self, target_indices: np.ndarray, weight: float = 12.0):
        """Inject discrete instantaneous spikes into target neurons."""
        idx = np.ascontiguousarray(target_indices, dtype=np.uint32)
        self._lib.chessfly_inject_spikes(
            self._handle,
            idx.ctypes.data_as(C.POINTER(C.c_uint32)),
            len(idx),
            C.c_float(weight),
        )

    def step(self, num_steps: int = 1, dt_ms: Optional[float] = None):
        """Step the simulation by `num_steps`."""
        dt = float(dt_ms if dt_ms is not None else self.dt_ms)
        self._lib.chessfly_step(self._handle, num_steps, C.c_float(dt))

    def simulate_window(
        self,
        duration_ms: float = 1000.0,
        dt_ms: Optional[float] = None,
        sensory_spike_events: Optional[Dict[int, List[int]]] = None,
        spike_weight: float = 15.0,
    ):
        """Simulate a full temporal window (default 1,000 ms = 1 second) from rest.

        sensory_spike_events: dictionary mapping time_step_idx -> list of target neuron indices.
        """
        dt = float(dt_ms if dt_ms is not None else self.dt_ms)
        total_steps = int(round(duration_ms / dt))

        if sensory_spike_events is None or not sensory_spike_events:
            # Continuous step
            self.step(total_steps, dt)
        else:
            # Advance step by step or in chunks delivering spike trains
            current_step = 0
            event_steps = sorted(sensory_spike_events.keys())
            for step_idx in event_steps:
                if step_idx >= total_steps:
                    break
                delta = step_idx - current_step
                if delta > 0:
                    self.step(delta, dt)
                    current_step = step_idx
                targets = np.array(sensory_spike_events[step_idx], dtype=np.uint32)
                self.inject_spikes(targets, spike_weight)
            # Remaining steps
            if total_steps > current_step:
                self.step(total_steps - current_step, dt)

    def get_motor_features(self, motor_indices: np.ndarray) -> np.ndarray:
        """Extract motor features: Feature_i = spikes_i + (Vm - V_rest) / 7.0 mV."""
        idx = np.ascontiguousarray(motor_indices, dtype=np.uint32)
        count = len(idx)
        out = np.zeros(count, dtype=np.float32)
        self._lib.chessfly_get_motor_features(
            self._handle,
            idx.ctypes.data_as(C.POINTER(C.c_uint32)),
            count,
            out.ctypes.data_as(C.POINTER(C.c_float)),
        )
        return out

    def get_all_spikes(self) -> np.ndarray:
        """Get spike counts for all neurons."""
        out = np.zeros(self.num_neurons, dtype=np.uint32)
        self._lib.chessfly_get_all_spikes(
            self._handle,
            out.ctypes.data_as(C.POINTER(C.c_uint32)),
        )
        return out

    def get_all_voltages(self) -> np.ndarray:
        """Get membrane potentials for all neurons."""
        out = np.zeros(self.num_neurons, dtype=np.float32)
        self._lib.chessfly_get_all_voltages(
            self._handle,
            out.ctypes.data_as(C.POINTER(C.c_float)),
        )
        return out

    def apply_plasticity(
        self,
        kc_indices: np.ndarray,
        mbon_indices: np.ndarray,
        dan_signal: float,
        eta: float = 0.001,
    ) -> int:
        """Apply dopamine-driven reinforcement learning across KC -> MBON synapses.

        dan_signal: +1.0 for PAM11 reward, -1.0 for PPL101 aversive
        """
        kcs = np.ascontiguousarray(kc_indices, dtype=np.uint32)
        mbons = np.ascontiguousarray(mbon_indices, dtype=np.uint32)
        modified = self._lib.chessfly_apply_plasticity(
            self._handle,
            kcs.ctypes.data_as(C.POINTER(C.c_uint32)),
            len(kcs),
            mbons.ctypes.data_as(C.POINTER(C.c_uint32)),
            len(mbons),
            C.c_float(dan_signal),
            C.c_float(eta),
        )
        return int(modified)

    def set_neuromodulation(self, octopamine: float = 0.0, conductance_gain: float = 1.0):
        """Configure octopaminergic excitation (attacks/tension) and conductance gain."""
        self._lib.chessfly_set_neuromodulation(
            self._handle,
            C.c_float(octopamine),
            C.c_float(conductance_gain),
        )
