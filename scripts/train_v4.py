"""Master Calibration Training Script for ChessFly V4 (Overclocked Titan Fly).

Trains the 1,409 descending motor readout projection matrix across an expanded
tactical, positional, prophylaxis, and endgame master dataset with dt = 0.25 ms.
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

# Titan Calibration Suite (Position FEN, Good Move, Bad Move, Target Good Score, Target Bad Score)
MASTER_CALIBRATION_SUITE_V4 = [
    # 1. Rook on Open File (1350 ELO)
    ("r1b2rk1/pp1nqppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10", "a1c1", "a2a3", 3.0, -1.0),
    # 2. Knight Outpost in Sicilian Dragon (1400 ELO)
    ("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 9", "d4b3", "g2g4", 2.5, -1.5),
    # 3. King Safety: Castle Kingside (1100 ELO)
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", "e1e2", 3.5, -4.0),
    # 4. Central Space Expansion: 1. d4 / 1. e4 (1000 ELO)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4", "h2h4", 2.5, -1.0),
    # 5. Free piece capture: Knight on e5 (850 ELO)
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", "f3e5", "h2h3", 4.0, -2.0),
    # 6. Queen Under Attack Escape (1000 ELO)
    ("r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2PPP/RNBQKB1R b kq - 1 6", "h4f6", "e8e7", 3.5, -5.0),
    # 7. Defend Attacked Bishop (1050 ELO)
    ("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", "b5a4", "b5d3", 3.0, -2.0),
    # 8. Pawn Promotion (1100 ELO)
    ("8/4P3/8/8/8/8/k7/4K3 w - - 0 1", "e7e8q", "e1d2", 4.5, -4.0),
    # 9. Interpose Check with Development (900 ELO)
    ("r1bqk1nr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4", "c1d2", "e1e2", 2.8, -4.5),
    # 10. Avoid Knight Fork (1150 ELO)
    ("r1b1kbnr/pppp1ppp/8/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1", "d2d4", "e5d7", 2.5, -3.5),
    # 11. Knight Evasion from Pawn Attack (1200 ELO)
    ("r1bqkbnr/pppp1ppp/2n5/1P2p3/4P3/8/P1PP1PPP/RNBQKBNR b KQkq - 0 3", "c6d4", "h7h6", 2.5, -3.0),
    # 12. Solid Opening Development (1150 ELO)
    ("r1b1kbnr/ppppqppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "b1c3", "f3e5", 2.8, -3.5),
    # 13. Queen's Gambit Exchange (1350 ELO)
    ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", "e7e6", "e7e5", 2.5, -2.5),
    # 14. French Defense solid center (1250 ELO)
    ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d2d4", "f1a6", 2.5, -3.0),
    # 15. Pinned piece exploitation (1300 ELO)
    ("r1bqkb1r/ppp2ppp/2np1n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4", "a7a6", "c6e4", 2.0, -2.5),
    # 16. Endgame King Centralization (1450 ELO)
    ("8/5k2/8/8/8/8/5K2/8 w - - 0 1", "f2e3", "f2g1", 3.0, -1.0),
    # 17. Scholar's Mate Delivery (750 ELO)
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", "h5f3", 5.0, -2.0),
    # 18. Prophylaxis: Kick Aggressive Knight on g4 (1800 ELO)
    ("r1bq1rk1/ppp2ppp/2np4/2b1p3/2B1P1n1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 7", "h2h3", "d3d4", 3.5, -3.0),
    # 19. Back-Rank Mate Emergency Defense (1350 ELO)
    ("4r1k1/5ppp/8/8/8/8/1R3PPP/6K1 w - - 0 1", "b2b1", "b2b3", 3.5, -5.0),
    # 20. Active Bishop Placement vs Passive Blocking (1850 ELO)
    ("r2q1rk1/pp1nbppp/2p1pn2/8/3P4/2NB1N2/PPP2PPP/R1BQ1RK1 w - - 2 9", "c1f4", "c1d2", 2.8, -1.5),
]


def train_v4():
    print("=" * 65)
    print("CHESSFLY V4 OVERCLOCKED TITAN FLY CALIBRATION TRAINING")
    print("=" * 65)

    graph = build_canonical_drosophila_connectome(num_total_neurons=8000, random_seed=42)
    dt_ms = 0.25
    base_rate_hz = 260.0
    sim = LifSimulator(graph.indptr, graph.indices, graph.weights, graph.num_neurons, dt_ms=dt_ms)
    encoder = SpatialRetinotopicEncoder(graph.circuits, rng_seed=42)
    neuromod = NeuromodulationController()
    readout = MotorReadoutLayer(graph.circuits)

    feature_matrix = []
    targets = []

    print(f"Extracting spatial features over {len(MASTER_CALIBRATION_SUITE_V4) * 2} training samples at dt={dt_ms}ms...")
    t0 = time.perf_counter()
    for fen, good_uci, bad_uci, good_score, bad_score in MASTER_CALIBRATION_SUITE_V4:
        board = chess.Board(fen)
        good_m = chess.Move.from_uci(good_uci)
        bad_m = chess.Move.from_uci(bad_uci)

        for m, score_val in [(good_m, good_score), (bad_m, bad_score)]:
            if m not in board.legal_moves:
                continue

            threat_map, pursuit_map, _ = encoder.extract_spatial_features(board, m)
            chem = neuromod.compute_state(board, m)
            events = encoder.generate_retinotopic_poisson_spikes(
                threat_map, pursuit_map, duration_ms=250.0, dt_ms=dt_ms, base_rate_hz=base_rate_hz
            )

            sim.reset()
            sim.set_neuromodulation(chem.octopamine, chem.conductance_gain)
            sim.simulate_window(250.0, dt_ms=dt_ms, sensory_spike_events=events, spike_weight=24.0)

            feats = readout.extract_motor_features(sim)
            feature_matrix.append(feats)
            targets.append(score_val)

    X = np.array(feature_matrix, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    elapsed_extract = time.perf_counter() - t0
    print(f"Feature extraction completed in {elapsed_extract:.2f}s. Matrix shape: {X.shape}")

    # Solve Ridge Regression: W = (X^T X + lambda * I)^(-1) X^T y
    print("Fitting Ridge regression motor projection layer...")
    reg_lambda = 0.05
    XTX = X.T @ X
    reg_matrix = reg_lambda * np.eye(XTX.shape[0], dtype=np.float32)
    XTy = X.T @ y
    W = np.linalg.solve(XTX + reg_matrix, XTy)

    readout.weights = W
    out_path = ROOT / "models" / "readout_v4.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    readout.save(out_path)
    print(f"Trained readout weights saved to: {out_path}")

    # Validation check on training positions
    print("\nVerifying training discrimination:")
    correct = 0
    total = 0
    for fen, good_uci, bad_uci, _, _ in MASTER_CALIBRATION_SUITE_V4:
        board = chess.Board(fen)
        good_m = chess.Move.from_uci(good_uci)
        bad_m = chess.Move.from_uci(bad_uci)

        def eval_move(mv):
            t_map, p_map, _ = encoder.extract_spatial_features(board, mv)
            chem = neuromod.compute_state(board, mv)
            ev = encoder.generate_retinotopic_poisson_spikes(t_map, p_map, 250.0, dt_ms, base_rate_hz)
            sim.reset()
            sim.set_neuromodulation(chem.octopamine, chem.conductance_gain)
            sim.simulate_window(250.0, dt_ms=dt_ms, sensory_spike_events=ev, spike_weight=24.0)
            f = readout.extract_motor_features(sim)
            return float(readout.compute_score(f))

        s_good = eval_move(good_m)
        s_bad = eval_move(bad_m)
        is_correct = s_good > s_bad
        if is_correct:
            correct += 1
        total += 1
        print(f"  {good_uci} ({s_good:+.2f}) vs {bad_uci} ({s_bad:+.2f}) -> {'PASS' if is_correct else 'FAIL'}")

    acc = (correct / total) * 100.0
    print(f"\nCalibration Accuracy: {correct}/{total} ({acc:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    train_v4()
