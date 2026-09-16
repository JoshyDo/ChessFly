"""Master Calibration Training Script for ChessFly V2.

Trains the 1,409 descending motor readout projection matrix on spatial retinotopic
features across tactical, positional, and pawn-structure master motifs.
"""

from pathlib import Path
import sys
import time
import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.simulator import LifSimulator
from chessfly.spatial_encoder import SpatialRetinotopicEncoder
from chessfly.neuromodulation import NeuromodulationController
from chessfly.motor_readout import MotorReadoutLayer

ROOT = Path(__file__).resolve().parents[1]

# Master Calibration Dataset (Position FEN, Best/Sound Move, Blunder/Mistake Move, Evaluation Target)
MASTER_CALIBRATION_SUITE = [
    # 1. Rook on Open File: 1. Rad1! vs 1. a3?
    ("r1b2rk1/pp1nqppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10", "a1d1", "a2a3", 2.0, -1.0),
    # 2. Knight Outpost: 1. Nd5! vs 1. h3?
    ("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 9", "d4b3", "g2g4", 1.8, -0.8),
    # 3. Castling / King Safety
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", "e1e2", 2.5, -3.0),
    # 4. Central Dominance (d4)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4", "h2h4", 2.0, -0.5),
    # 5. Free piece capture (tactical opportunism)
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", "f3e5", "h2h3", 3.5, -1.0),
    # 6. Absolute Pin Exploit: White pins knight to King
    ("r1bqk1nr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "g8f6", "c6d4", 1.5, -2.0),
    # 7. Passed Pawn Push (6th rank)
    ("8/4P3/8/8/8/8/k7/4K3 w - - 0 1", "e7e8q", "e1d2", 4.0, -3.5),
    # 8. Queen Threat Escape: 6... Qf6! vs 6... Ke7??
    ("r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2PPP/RNBQKB1R b kq - 1 6", "h4f6", "e8e7", 3.0, -5.0),
    # 9. Avoid Knight Fork
    ("r1b1kbnr/pppp1ppp/8/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1", "d2d4", "e5d7", 2.0, -3.5),
    # 10. Interpose check with piece development
    ("r1bqk1nr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4", "c1d2", "e1e2", 2.2, -4.0),
    # 11. Defend attacked Bishop
    ("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", "b5a4", "d2d3", 2.5, -3.0),
    # 12. Ponziani Solid development
    ("r1b1kbnr/ppppqppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "b1c3", "f3e5", 2.0, -4.0),
]


def train_v2():
    print("=" * 65)
    print("CHESSFLY V2 GRANDMASTER CALIBRATION TRAINING")
    print("=" * 65)

    graph = build_canonical_drosophila_connectome(num_total_neurons=8000, random_seed=42)
    sim = LifSimulator(graph.indptr, graph.indices, graph.weights, graph.num_neurons, dt_ms=0.5)
    encoder = SpatialRetinotopicEncoder(graph.circuits)
    neuromod = NeuromodulationController()
    readout = MotorReadoutLayer(graph.circuits)

    feature_matrix = []
    targets = []

    print(f"Extracting spatial retinotopic features over {len(MASTER_CALIBRATION_SUITE) * 2} training samples...")
    for fen, good_uci, bad_uci, good_score, bad_score in MASTER_CALIBRATION_SUITE:
        board = chess.Board(fen)
        good_m = chess.Move.from_uci(good_uci)
        bad_m = chess.Move.from_uci(bad_uci)

        for m, score_val in [(good_m, good_score), (bad_m, bad_score)]:
            if m not in board.legal_moves:
                continue

            threat_map, pursuit_map, _ = encoder.extract_spatial_features(board, m)
            chem = neuromod.compute_state(board, m)
            events = encoder.generate_retinotopic_poisson_spikes(
                threat_map, pursuit_map, duration_ms=250.0, dt_ms=0.5
            )

            sim.reset()
            sim.set_neuromodulation(chem.octopamine, chem.conductance_gain)
            sim.simulate_window(250.0, dt_ms=0.5, sensory_spike_events=events, spike_weight=20.0)

            feats = readout.extract_motor_features(sim)
            feature_matrix.append(feats)
            targets.append(score_val)

    X = np.array(feature_matrix, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    print(f"Training Ridge Regression projection on {X.shape[0]} samples across {X.shape[1]} DN features...")
    readout.train_projection(X, y, regularization_lambda=0.02)

    out_file = ROOT / "models" / "readout_v2.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    readout.save(out_file)
    print(f"V2 Model saved successfully to: {out_file}!")


if __name__ == "__main__":
    train_v2()
