"""ChessFly Biological Connectome Engine.

Strict 1-ply position evaluation and move selection executed entirely by
simulated Drosophila melanogaster nervous system dynamics.
Zero external search trees, minimax, MCTS, or chess engine helpers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import chess
import numpy as np

from chessfly.connectome_loader import ConnectomeGraph, build_canonical_drosophila_connectome
from chessfly.circuit_registry import CircuitRegistry
from chessfly.simulator import LifSimulator
from chessfly.chess_encoder import ChessSensoryEncoder
from chessfly.motor_readout import MotorReadoutLayer


class ChessFlyEngine:
    """Pure biological connectome chess engine."""

    def __init__(
        self,
        connectome_graph: Optional[ConnectomeGraph] = None,
        weights_path: Optional[Path] = None,
        simulation_duration_ms: float = 1000.0,
        dt_ms: float = 0.5,
        spike_weight: float = 20.0,
        base_rate_hz: float = 180.0,
    ):
        if connectome_graph is None:
            # Generate canonical Drosophila connectome with exact circuits
            self.graph = build_canonical_drosophila_connectome(num_total_neurons=10000)
        else:
            self.graph = connectome_graph

        self.circuits = self.graph.circuits
        self.simulation_duration_ms = float(simulation_duration_ms)
        self.dt_ms = float(dt_ms)
        self.spike_weight = float(spike_weight)
        self.base_rate_hz = float(base_rate_hz)

        # Initialize biophysical LIF simulator backed by the Rust solver
        self.sim = LifSimulator(
            indptr=self.graph.indptr,
            indices=self.graph.indices,
            weights=self.graph.weights,
            num_neurons=self.graph.num_neurons,
            dt_ms=self.dt_ms,
        )

        # Initialize sensory encoder and motor readout layer
        self.encoder = ChessSensoryEncoder(self.circuits)
        if weights_path and Path(weights_path).exists():
            self.readout = MotorReadoutLayer.load(weights_path, self.circuits)
        else:
            self.readout = MotorReadoutLayer(self.circuits)

    def evaluate_candidate_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single candidate move through a 1,000 ms connectome simulation from rest.

        Returns:
            move_score (float): preference score from descending motor readout
            diagnostics (dict): firing rates and biophysical telemetry
        """
        # 1. Extract visual features (looming threat vs target pursuit)
        threat_intensity, pursuit_intensity, metrics = self.encoder.extract_features(board, move)

        # 2. Generate 180 Hz Poisson-style spike trains
        spike_events = self.encoder.generate_poisson_spike_trains(
            threat_intensity=threat_intensity,
            pursuit_intensity=pursuit_intensity,
            duration_ms=self.simulation_duration_ms,
            dt_ms=self.dt_ms,
            base_rate_hz=self.base_rate_hz,
        )

        # 3. Reset network state back to resting potential (-52 mV)
        self.sim.reset()

        # 4. Run 1,000 ms simulation window from rest
        self.sim.simulate_window(
            duration_ms=self.simulation_duration_ms,
            dt_ms=self.dt_ms,
            sensory_spike_events=spike_events,
            spike_weight=self.spike_weight,
        )

        # 5. Extract features from ~1,409 descending motor neurons:
        # Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV
        motor_features = self.readout.extract_motor_features(self.sim)

        # 6. Linear projection readout
        score = self.readout.compute_score(motor_features)

        # Collect diagnostics
        escape_indices = self.circuits.escape_readout_neurons
        approach_indices = self.circuits.approach_readout_neurons

        escape_activity = float(np.mean(self.sim.get_motor_features(escape_indices))) if len(escape_indices) else 0.0
        approach_activity = float(np.mean(self.sim.get_motor_features(approach_indices))) if len(approach_indices) else 0.0

        diagnostics = {
            "score": score,
            "threat_intensity": threat_intensity,
            "pursuit_intensity": pursuit_intensity,
            "escape_pathway_activity": escape_activity,
            "approach_pathway_activity": approach_activity,
            "hanging_material": metrics.get("hanging_material", 0.0),
            "captured_val": metrics.get("captured_val", 0.0),
        }

        return score, diagnostics

    def select_best_move(
        self,
        board: chess.Board,
        candidate_moves: Optional[List[chess.Move]] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[chess.Move], float, Dict[chess.Move, Dict[str, float]]]:
        """Perform 1-ply position evaluation across all legal moves and select best move."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, {}

        if candidate_moves is not None:
            # Filter to legal subset
            eval_moves = [m for m in candidate_moves if m in legal_moves]
            if not eval_moves:
                eval_moves = legal_moves
        else:
            eval_moves = legal_moves

        best_move = None
        best_score = -float("inf")
        move_evals: Dict[chess.Move, Dict[str, float]] = {}

        t0 = time.time()
        for move in eval_moves:
            score, diag = self.evaluate_candidate_move(board, move)
            move_evals[move] = diag

            if verbose:
                print(
                    f"Move {move.uci()}: Score={score:+.2f} | "
                    f"Threat={diag['threat_intensity']:.2f} (Escape={diag['escape_pathway_activity']:.1f}) | "
                    f"Pursuit={diag['pursuit_intensity']:.2f} (Approach={diag['approach_pathway_activity']:.1f})"
                )

            if score > best_score:
                best_score = score
                best_move = move

        duration = time.time() - t0
        if verbose:
            print(f"Evaluated {len(eval_moves)} candidate moves in {duration:.2f}s. Best: {best_move} ({best_score:+.2f})")

        return best_move, best_score, move_evals
