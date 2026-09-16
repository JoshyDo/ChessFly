"""Unit and benchmark tests for ChessFly V2 (Grandmaster Fly)."""

from pathlib import Path
import chess
import numpy as np
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.spatial_encoder import SpatialRetinotopicEncoder
from chessfly.neuromodulation import NeuromodulationController
from chessfly.engine_v2 import ChessFlyV2Engine

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_V2 = ROOT / "models" / "readout_v2.npz"


@pytest.fixture(scope="function")
def v2_engine():
    graph = build_canonical_drosophila_connectome(num_total_neurons=5000, random_seed=42)
    eng = ChessFlyV2Engine(
        connectome_graph=graph,
        weights_path=WEIGHTS_V2 if WEIGHTS_V2.exists() else None,
        simulation_duration_ms=250.0,
        dt_ms=0.5,
    )
    return eng


def test_spatial_retinotopic_partitioning():
    """Verify that all 64 squares receive allocated ommatidial visual columns."""
    graph = build_canonical_drosophila_connectome(num_total_neurons=5000, random_seed=42)
    encoder = SpatialRetinotopicEncoder(graph.circuits)

    assert len(encoder.square_to_lplc2) == 64
    assert len(encoder.square_to_lc4) == 64
    assert len(encoder.square_to_lc10a) == 64
    assert len(encoder.square_to_lc9) == 64

    # Total allocated neurons across the 64 columns must equal total circuit neurons
    total_lplc2 = sum(len(cols) for cols in encoder.square_to_lplc2)
    assert total_lplc2 == len(graph.circuits.lplc2)


def test_ray_tracing_pin_detection():
    """Verify that pinned friendly pieces receive elevated localized threat."""
    graph = build_canonical_drosophila_connectome(num_total_neurons=4000, random_seed=42)
    encoder = SpatialRetinotopicEncoder(graph.circuits)

    # Position where Black Knight on c6 is absolutely pinned to King on e8 by White Bishop on b5
    board = chess.Board("r1bqkb1r/ppp2ppp/2np1n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
    move = chess.Move.from_uci("a7a6")

    threat_map, pursuit_map, _ = encoder.extract_spatial_features(board, move)

    # Square c6 has the pinned knight
    assert threat_map[chess.C6] > 0.15, f"Pinned knight on c6 must register threat, got {threat_map[chess.C6]}"


def test_neuromodulation_octopamine_surge():
    """Giving check or attacking opposing king must trigger octopaminergic surge."""
    controller = NeuromodulationController()

    # Black Queen delivers check on h4+
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    check_move = chess.Move.from_uci("d8h4")

    state = controller.compute_state(board, check_move)
    assert state.octopamine >= 0.50, f"Delivering check must surge octopamine >= 0.5, got {state.octopamine}"
    assert state.conductance_gain > 1.2, "Synaptic gain must be amplified during octopaminergic surge"


# Grandmaster-tier benchmark puzzles
MASTER_PUZZLES = [
    # 1. Queen Under Attack: save Queen (h4)
    ("r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2PPP/RNBQKB1R b kq - 1 6", ["h4f6", "h4e7", "h4h5", "h4d8", "h4g4"]),
    # 2. King Safety: Castle kingside (O-O)
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e1g1"]),
    # 3. Clean capture of hanging knight
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", ["f3e5"]),
    # 4. Promote passed pawn on 7th rank
    ("8/4P3/8/8/8/8/k7/4K3 w - - 0 1", ["e7e8q", "e7e8r"]),
    # 5. Defend attacked bishop
    ("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", ["b5a4", "b5c6", "b5c4"]),
    # 6. Interpose check
    ("r1bqk1nr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4", ["c1d2", "c2c3", "b1d2", "b1c3"]),
]


@pytest.mark.parametrize("fen, expected_moves", MASTER_PUZZLES)
def test_v2_master_puzzle_accuracy(v2_engine, fen: str, expected_moves: list):
    """Verify ChessFly V2 solves master tactical and positional motifs accurately."""
    board = chess.Board(fen)
    best_move, score, _ = v2_engine.select_best_move(board)

    assert best_move is not None
    best_uci = best_move.uci()
    print(f"\nFEN: {fen} -> Selected: {best_uci} (Expected: {expected_moves})")

    assert best_uci in expected_moves, (
        f"V2 failed puzzle! Selected {best_uci}, expected one of {expected_moves}"
    )
