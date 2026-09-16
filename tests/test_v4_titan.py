"""Unit and benchmark tests for ChessFly V4 (Overclocked Titan Fly)."""

from pathlib import Path
import chess
import pytest

from chessfly.engine_v4 import ChessFlyV4Engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="function")
def v4_engine():
    return ChessFlyV4Engine()


def test_v4_canonical_architecture(v4_engine):
    """Verify V4 connectome scaling and circuit dimensions."""
    assert v4_engine.graph.num_neurons == 8000
    assert len(v4_engine.circuits.descending_neurons) == 1409
    assert len(v4_engine.encoder.square_to_lplc2) == 64
    assert len(v4_engine.encoder.square_to_lc10a) == 64


def test_v4_deterministic_reproducibility(v4_engine):
    """Verify that move evaluation is 100% deterministic across multiple runs."""
    board = chess.Board("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    move1, score1, _ = v4_engine.select_best_move(board)
    move2, score2, _ = v4_engine.select_best_move(board)

    assert move1 == move2
    assert abs(score1 - score2) < 1e-5


def test_v4_checkmate_delivery(v4_engine):
    """Verify V4 delivers Scholar's Mate cleanly."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    best_move, _, _ = v4_engine.select_best_move(board)
    assert best_move == chess.Move.from_uci("h5f7")


def test_v4_emergency_back_rank_escape(v4_engine):
    """Verify V4 escapes back-rank checkmate."""
    board = chess.Board("4r1k1/5ppp/8/8/8/8/1R3PPP/6K1 w - - 0 1")
    best_move, _, _ = v4_engine.select_best_move(board)
    assert best_move.uci() in ["b2b8", "b2b1", "h2h3", "g1f1"]


TITAN_SAMPLE_PUZZLES = [
    # Clean capture of hanging piece
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", ["f3e5"]),
    # Castle kingside
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e1g1", "b1c3"]),
    # Prophylaxis: Kick Ng4
    ("r1bq1rk1/ppp2ppp/2np4/2b1p3/2B1P1n1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 7", ["h2h3", "c1g5"]),
    # Carlsbad mobilization
    ("r1b2rk1/ppqn1ppp/2p1pn2/3p4/2PP4/1PNBPN2/P4PPP/R2Q1RK1 w - - 0 11", ["b3b4", "a1c1", "c4d5", "d1c2", "f1e1"]),
]


@pytest.mark.parametrize("fen, expected_moves", TITAN_SAMPLE_PUZZLES)
def test_v4_sample_puzzles(v4_engine, fen: str, expected_moves: list):
    """Verify V4 solves tactical and strategic puzzles."""
    board = chess.Board(fen)
    best_move, _, _ = v4_engine.select_best_move(board)
    assert best_move is not None
    assert best_move.uci() in expected_moves
