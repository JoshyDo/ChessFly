"""Unit and benchmark tests for ChessFly V3 (Super-Grandmaster Fly)."""

from pathlib import Path
import chess
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.see_optics import static_exchange_evaluation, evaluate_move_see
from chessfly.opening_memory import MushroomBodyOpeningMemory
from chessfly.engine_v3 import ChessFlyV3Engine

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_V3 = ROOT / "models" / "readout_v3.npz"


@pytest.fixture(scope="module")
def v3_engine():
    graph = build_canonical_drosophila_connectome(num_total_neurons=5000, random_seed=42)
    eng = ChessFlyV3Engine(
        connectome_graph=graph,
        weights_path=WEIGHTS_V3 if WEIGHTS_V3.exists() else None,
        simulation_duration_ms=250.0,
        dt_ms=0.5,
    )
    return eng


def test_static_exchange_evaluation_accuracy():
    """Verify that SEE optics detects clean captures vs losing trades."""
    # 1. Clean capture of hanging knight: Nxe5
    b1 = chess.Board("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    m1 = chess.Move.from_uci("f3e5")
    see1 = evaluate_move_see(b1, m1)
    assert see1 > 250, f"Free knight capture must yield SEE > 250 cp, got {see1}"

    # 2. Defended exchange: Knight takes pawn on e5 defended by Nc6
    b2 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    m2 = chess.Move.from_uci("f3e5")
    see2 = evaluate_move_see(b2, m2)
    assert see2 < 0, f"Losing knight for pawn must yield negative SEE, got {see2}"


def test_mushroom_body_opening_memory():
    """Verify that Kenyon Cell -> MBON associative memory activates for GM opening theory."""
    memory = MushroomBodyOpeningMemory(memory_strength=300.0)

    # In initial position, 1. e4 and 1. d4 should have strong associative bias
    b0 = chess.Board()
    e4_move = chess.Move.from_uci("e2e4")
    h4_move = chess.Move.from_uci("h2h4")

    assert memory.get_memory_bias(b0, e4_move) > 100.0
    assert memory.get_memory_bias(b0, h4_move) == 0.0

    # Ruy Lopez 3... a6
    b_rl = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    a6_move = chess.Move.from_uci("a7a6")
    assert memory.get_memory_bias(b_rl, a6_move) > 100.0


SUPER_GM_PUZZLES = [
    # 1. Queen Under Attack: save Queen (h4)
    ("r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2PPP/RNBQKB1R b kq - 1 6", ["h4f6", "h4e7", "h4h5", "h4d8", "h4g4"]),
    # 2. King Safety: Castle kingside (O-O)
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e1g1"]),
    # 3. Clean capture of hanging knight (SEE verified)
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", ["f3e5"]),
    # 4. Promote passed pawn on 7th rank
    ("8/4P3/8/8/8/8/k7/4K3 w - - 0 1", ["e7e8q", "e7e8r"]),
    # 5. Defend attacked bishop in Ruy Lopez
    ("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", ["b5a4", "b5c6", "b5c4"]),
    # 6. Interpose check with piece development
    ("r1bqk1nr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4", ["c1d2", "c2c3", "b1d2", "b1c3"]),
    # 7. French Defense central strike
    ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["d2d4"]),
    # 8. Knight Outpost
    ("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 9", ["d4b3", "c3d5", "d1d2", "f1c4"]),
]


@pytest.mark.parametrize("fen, expected_moves", SUPER_GM_PUZZLES)
def test_v3_supergm_puzzle_accuracy(v3_engine, fen: str, expected_moves: list):
    """Verify ChessFly V3 solves candidate master tactical and positional motifs."""
    board = chess.Board(fen)
    best_move, score, _ = v3_engine.select_best_move(board)

    assert best_move is not None
    best_uci = best_move.uci()
    print(f"\nFEN: {fen} -> Selected: {best_uci} (Expected: {expected_moves})")

    assert best_uci in expected_moves, (
        f"V3 failed puzzle! Selected {best_uci}, expected one of {expected_moves}"
    )
