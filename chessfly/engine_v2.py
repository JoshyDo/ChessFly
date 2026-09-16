"""ChessFly V2: Grandmaster Biological Connectome Engine.

Combines:
1. 64-Square Spatial Retinotopy (ommatidial receptive fields)
2. Ray-Tracing Threat Optics (pins, skewers, passed pawns, knight outposts, open files)
3. Central Complex Neuromodulation (Octopaminergic arousal & Serotonergic vigilance)
4. 1,409 Descending Motor Neuron Readout
Strict 1-ply biological connectome simulation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import chess
import numpy as np

from chessfly.connectome_loader import ConnectomeGraph, build_canonical_drosophila_connectome
from chessfly.simulator import LifSimulator
from chessfly.spatial_encoder import SpatialRetinotopicEncoder
from chessfly.neuromodulation import NeuromodulationController
from chessfly.motor_readout import MotorReadoutLayer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_WEIGHTS = ROOT / "models" / "readout_v2.npz"


class ChessFlyV2Engine:
    """Grandmaster-tier biological connectome chess engine."""

    def __init__(
        self,
        connectome_graph: Optional[ConnectomeGraph] = None,
        weights_path: Optional[Path] = None,
        simulation_duration_ms: float = 400.0,
        dt_ms: float = 0.5,
        spike_weight: float = 22.0,
        base_rate_hz: float = 180.0,
    ):
        if connectome_graph is None:
            self.graph = build_canonical_drosophila_connectome(num_total_neurons=10000, random_seed=42)
        else:
            self.graph = connectome_graph

        self.circuits = self.graph.circuits
        self.simulation_duration_ms = float(simulation_duration_ms)
        self.dt_ms = float(dt_ms)
        self.spike_weight = float(spike_weight)
        self.base_rate_hz = float(base_rate_hz)

        # Biophysical LIF solver
        self.sim = LifSimulator(
            indptr=self.graph.indptr,
            indices=self.graph.indices,
            weights=self.graph.weights,
            num_neurons=self.graph.num_neurons,
            dt_ms=self.dt_ms,
        )

        # Retinotopic spatial encoder & neuromodulation controller
        self.encoder = SpatialRetinotopicEncoder(self.circuits)
        self.neuromod = NeuromodulationController()

        # Load or initialize motor readout layer
        w_path = weights_path or (DEFAULT_V2_WEIGHTS if DEFAULT_V2_WEIGHTS.exists() else None)
        if w_path and Path(w_path).exists():
            self.readout = MotorReadoutLayer.load(Path(w_path), self.circuits)
        else:
            self.readout = MotorReadoutLayer(self.circuits)

    def evaluate_candidate_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate move 1-ply through spatial retinotopic connectome simulation."""
        # 1. Spatial Retinotopic Feature Mapping (64 ommatidial receptive fields)
        threat_map, pursuit_map, metrics = self.encoder.extract_spatial_features(board, move)

        # 2. Neuromodulation (Octopamine / Serotonin)
        chem_state = self.neuromod.compute_state(board, move)

        # 3. Generate localized Poisson spike trains
        spike_events = self.encoder.generate_retinotopic_poisson_spikes(
            threat_map=threat_map,
            pursuit_map=pursuit_map,
            duration_ms=self.simulation_duration_ms,
            dt_ms=self.dt_ms,
            base_rate_hz=self.base_rate_hz,
        )

        # 4. Reset network to rest (-52 mV)
        self.sim.reset()

        # 5. Apply neuromodulation to Rust solver
        self.sim.set_neuromodulation(
            octopamine=chem_state.octopamine,
            conductance_gain=chem_state.conductance_gain,
        )

        # 6. Run recurrent central complex simulation
        self.sim.simulate_window(
            duration_ms=self.simulation_duration_ms,
            dt_ms=self.dt_ms,
            sensory_spike_events=spike_events,
            spike_weight=self.spike_weight,
        )

        # 7. Motor Readout: Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV
        motor_features = self.readout.extract_motor_features(self.sim)

        # 8. Linear projection scoring
        score = self.readout.compute_score(motor_features)

        escape_indices = self.circuits.escape_readout_neurons
        approach_indices = self.circuits.approach_readout_neurons
        escape_act = float(np.mean(self.sim.get_motor_features(escape_indices))) if len(escape_indices) else 0.0
        approach_act = float(np.mean(self.sim.get_motor_features(approach_indices))) if len(approach_indices) else 0.0

        diagnostics = {
            "score": score,
            "mean_threat": metrics.get("mean_threat", 0.0),
            "mean_pursuit": metrics.get("mean_pursuit", 0.0),
            "octopamine": chem_state.octopamine,
            "escape_activity": escape_act,
            "approach_activity": approach_act,
        }
        return score, diagnostics

    def select_best_move(
        self,
        board: chess.Board,
        candidate_moves: Optional[List[chess.Move]] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[chess.Move], float, Dict[chess.Move, Dict[str, float]]]:
        """Evaluate all legal moves 1-ply and select optimal move."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, {}

        eval_moves = [m for m in candidate_moves if m in legal_moves] if candidate_moves else legal_moves
        if not eval_moves:
            eval_moves = legal_moves

        best_move = None
        best_score = -float("inf")
        move_evals: Dict[chess.Move, Dict[str, float]] = {}

        t0 = time.perf_counter()
        for move in eval_moves:
            score, diag = self.evaluate_candidate_move(board, move)
            move_evals[move] = diag

            if verbose:
                print(f"Move {move.uci()}: Score={score:+.2f} | OA={diag['octopamine']:.2f} | Escape={diag['escape_activity']:.1f} | Approach={diag['approach_activity']:.1f}")

            if score > best_score:
                best_score = score
                best_move = move

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"V2 Evaluated {len(eval_moves)} moves in {elapsed:.2f}s. Best: {best_move} ({best_score:+.2f})")

        return best_move, best_score, move_evals
