"""Tactical benchmark test suite: Verifies zero single-move piece blunders in ChessFly."""

from pathlib import Path
from typing import List, Tuple
import chess
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.engine import ChessFlyEngine

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = ROOT / "models" / "readout_weights.npz"


@pytest.fixture(scope="module")
def engine():
    # Use canonical connectome with trained readout weights
    graph = build_canonical_drosophila_connectome(num_total_neurons=6000, random_seed=42)
    weights_path = WEIGHTS_FILE if WEIGHTS_FILE.exists() else None
    eng = ChessFlyEngine(
        connectome_graph=graph,
        weights_path=weights_path,
        simulation_duration_ms=500.0,
        dt_ms=0.5,
    )
    return eng


# Suite of tactical positions with explicit Blunder moves and Sound moves
BLUNDER_BENCHMARKS = [
    # 1. Defend Queen under attack: Black Queen on h4 is attacked by White Knight on f3
    # Blunder: 6... Ke7?? (hangs Queen)
    # Sound: 6... Qh5 or 6... Qe7
    (
        "r1b1k2r/pppp1ppp/2n5/4p3/4P2q/3P1N2/PPP2KPP/RNBQ1B1R b kq - 1 6",
        ["h4h5", "h4e7", "h4f6", "h4d8"],
        ["e8e7", "a7a6", "h7h6"],
    ),
    # 2. Free piece capture: White can capture free Black Knight on e5
    # Blunder: 4. h3?? (ignores capture, leaves Knight)
    # Sound: 4. Nxe5 (clean piece capture)
    (
        "r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
        ["f3e5"],
        ["h2h3", "a2a3", "g2g3"],
    ),
    # 3. Hanging Bishop: White Bishop on b5 is attacked by pawn a6
    # Blunder: White plays 4. d3?? (hangs Bishop to axb5)
    # Sound: 4. Ba4 or 4. Bxc6
    (
        "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        ["b5a4", "b5c6", "b5c4"],
        ["d2d3", "h2h3", "g2g3"],
    ),
    # 4. Don't blunder Queen into attack: White to move, don't play 3. Qxf7+?? (Kxf7)
    # Position after 1. e4 e5 2. Qh5 Nc6 3. Bc4 g6
    (
        "r1bqkbnr/pppp1p1p/2n3p1/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
        ["h5f3", "h5e2", "h5d1"],
        ["h5h7", "c4f7"],
    ),
    # 5. Hanging piece blunder in Giuoco Piano: White plays 4. Nxe5?? (hangs Knight to 4... Nxe5)
    (
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ["d2d3", "c2c3", "e1g1", "b1c3"],
        ["f3e5"],
    ),
    # 6. Hanging Knight to Queen: White plays 3. Nxe5?? (hangs Knight to 3... Qxe5)
    (
        "r1b1kbnr/ppppqppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ["b1c3", "d2d3", "f1c4"],
        ["f3e5"],
    ),
    # 7. Don't blunder Knight into capture:
    # White Knight on e5: 1. Nxd7?? or 1. Nxf7 vs sound moves
    (
        "r1b1kbnr/pppp1ppp/8/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
        ["d2d4", "e2e4", "e5f3", "b1c3"],
        ["e5d7"],
    ),
    # 8. Defend hanging Knight on c6 attacked by pawn b5
    (
        "r1bqkbnr/pppp1ppp/2n5/1P2p3/4P3/8/P1PP1PPP/RNBQKBNR b KQkq - 0 3",
        ["c6d4", "c6e7", "c6a5", "c6b8"],
        ["a7a6", "h7h6", "d7d6"],
    ),
]


@pytest.mark.parametrize("fen, sound_moves, blunder_moves", BLUNDER_BENCHMARKS)
def test_zero_single_move_piece_blunders(engine, fen: str, sound_moves: List[str], blunder_moves: List[str]):
    """Verify that ChessFly connectome move evaluation strictly rejects single-move piece blunders."""
    board = chess.Board(fen)

    best_move, best_score, evals = engine.select_best_move(board)
    assert best_move is not None, f"Engine must select a legal move in position {fen}"

    best_uci = best_move.uci()
    print(f"\nFEN: {fen}")
    print(f"Selected Move: {best_uci} (Score: {best_score:+.2f})")

    # 1. Absolute rule: Engine must NEVER select an explicitly identified single-move piece blunder!
    assert best_uci not in blunder_moves, (
        f"BLUNDER DETECTED! Engine selected {best_uci} in position {fen}!"
    )

    # 2. Check that Giant Fiber escape pathway penalizes blunder moves
    for blunder_uci in blunder_moves:
        blunder_m = chess.Move.from_uci(blunder_uci)
        if blunder_m in evals:
            blunder_diag = evals[blunder_m]
            # Blunder must trigger high looming threat or escape activity
            assert blunder_diag["threat_intensity"] > 0.1 or blunder_diag["hanging_material"] > 0, (
                f"Blunder {blunder_uci} was not flagged with looming threat!"
            )
            # Favorable / best score must be higher than blunder score
            assert best_score > blunder_diag["score"], (
                f"Best move score {best_score} must be strictly higher than blunder score {blunder_diag['score']}"
            )
