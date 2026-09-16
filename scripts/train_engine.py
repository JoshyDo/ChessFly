"""Training script for ChessFly: Optimizes projection matrix and dopaminergic plasticity."""

from pathlib import Path
from typing import List, Tuple
import sys
import time
import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.simulator import LifSimulator
from chessfly.chess_encoder import ChessSensoryEncoder
from chessfly.dopamine_rl import DopamineRLTrainer
from chessfly.motor_readout import MotorReadoutLayer


TRAINING_FENS = [
    # 1. Hanging Queen blunder vs retreat
    ("r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2KPP/RNBQ1B1R w kq - 1 6", "f3h4", "f2e2"),
    # 2. Hanging Knight capture
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "d2d4", "f3e5"),
    # 3. Defend hanging bishop
    ("rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 3", "c1d2", "d1g4"),
    # 4. Fork threat avoidance (Black knight forks King and Rook)
    ("r1bqk2r/pppp1ppp/2n5/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 1 6", "c1d2", "a2a3"),
    # 5. Clean piece capture (White captures free knight on e5)
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", "f3e5", "h2h3"),
    # 6. Pawn promotion capture
    ("8/4P3/8/8/8/8/k7/4K3 w - - 0 1", "e7e8q", "e1d2"),
    # 7. Check avoidance (King under check must move or block)
    ("rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3", "c1d2", "c2c3"),
    # 8. Central space expansion (e4 / d4)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "h2h4"),
]


def generate_training_data(
    num_samples: int = 100,
) -> List[Tuple[chess.Board, chess.Move, float]]:
    """Generate training samples with target evaluations (+1.0 sound, -1.0 blunder)."""
    dataset = []
    for fen, good_uci, bad_uci in TRAINING_FENS:
        board = chess.Board(fen)
        good_m = chess.Move.from_uci(good_uci)
        bad_m = chess.Move.from_uci(bad_uci)
        if good_m in board.legal_moves:
            dataset.append((board.copy(), good_m, 1.5))
        if bad_m in board.legal_moves:
            dataset.append((board.copy(), bad_m, -2.5))
    return dataset


def train_chessfly():
    print("Initializing Drosophila canonical connectome...")
    graph = build_canonical_drosophila_connectome(num_total_neurons=8000)
    sim = LifSimulator(graph.indptr, graph.indices, graph.weights, graph.num_neurons, dt_ms=0.5)
    encoder = ChessSensoryEncoder(graph.circuits)
    readout = MotorReadoutLayer(graph.circuits)
    rl_trainer = DopamineRLTrainer(sim, graph.circuits, encoder, learning_rate_eta=0.005)

    dataset = generate_training_data()
    print(f"Generated {len(dataset)} calibration positions.")

    # 1. Dopaminergic plasticity training
    print("Running dopamine-driven KC->MBON plasticity updates...")
    plastic_updates = 0
    for board, move, _ in dataset:
        mod = rl_trainer.train_single_move(board, move, simulation_duration_ms=200.0, dt_ms=0.5)
        plastic_updates += mod
    print(f"Total synaptic connections updated via dopamine: {plastic_updates}")

    # 2. Extract feature matrix for motor readout layer
    print("Extracting descending motor features across ~1,409 DNs...")
    feature_matrix = []
    targets = []

    for idx, (board, move, target_val) in enumerate(dataset):
        threat, pursuit, _ = encoder.extract_features(board, move)
        events = encoder.generate_poisson_spike_trains(threat, pursuit, duration_ms=200.0, dt_ms=0.5)

        sim.reset()
        sim.simulate_window(200.0, dt_ms=0.5, sensory_spike_events=events, spike_weight=18.0)
        feats = readout.extract_motor_features(sim)

        feature_matrix.append(feats)
        targets.append(target_val)

    X = np.array(feature_matrix, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    # 3. Train linear projection layer
    print("Fitting regularized ridge projection matrix...")
    readout.train_projection(X, y, regularization_lambda=0.05)

    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(exist_ok=True)
    out_file = model_dir / "readout_weights.npz"
    readout.save(out_file)
    print(f"Model saved successfully to {out_file}!")


if __name__ == "__main__":
    train_chessfly()
