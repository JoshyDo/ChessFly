"""Mushroom Body Associative Opening Memory (ChessFly V3).

Biophysical analogy: Long-term associative memory in Drosophila Kenyon Cells (KCs)
and Mushroom Body Output Neurons (MBONs).

Grandmaster opening principles (Italian, Ruy Lopez, Sicilian, French, Caro-Kann,
Queen's Gambit, King's Indian) are imprinted as associative synaptic patterns in
the mushroom body. When a board state matches an imprinted odor/pattern,
KCs excite approach MBONs to guide early development without tree search.
"""

from typing import Dict, List, Optional
import chess

# Canonical Grandmaster Openings Book indexed by EPD (board + turn + castling + ep)
GM_OPENING_BOOK: Dict[str, List[str]] = {
    # 1. Initial position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": ["e2e4", "d2d4", "c2c4", "g1f3"],
    # 1. e4 replies
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -": ["c7c5", "e7e5", "e7e6", "c7c6", "g8f6"],
    # 1. d4 replies
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -": ["d7d5", "g8f6", "e7e6", "c7c5"],
    # Open Game: 1. e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["g1f3", "b1c3", "f1c4"],
    # Sicilian Defense: 1. e4 c5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["g1f3", "b1c3", "c2c3"],
    # French Defense: 1. e4 e6
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["d2d4"],
    # Caro-Kann: 1. e4 c6
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["d2d4"],
    # 1. e4 e5 2. Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": ["b8c6", "g8f6", "d7d6"],
    # 1. e4 e5 2. Nf3 Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": ["f1b5", "f1c4", "d2d4", "b1c3"],
    # Ruy Lopez: 1. e4 e5 2. Nf3 Nc6 3. Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": ["a7a6", "g8f6", "d7d6"],
    # Ruy Lopez: 3... a6 response
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -": ["b5a4", "b5c6", "b5c4"],
    # Italian Game: 1. e4 e5 2. Nf3 Nc6 3. Bc4
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": ["f1c5", "g8f6", "d7d6"],
    # Italian Game: 3... Bc5
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -": ["e1g1", "c2c3", "d2d3", "b1c3"],
    # Interpose check on d4/e4
    "r1bqk1nr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq -": ["c1d2", "c2c3", "b1d2", "b1c3"],
    # Queen's Gambit: 1. d4 d5 2. c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -": ["e7e6", "c7c6", "d5c4", "g8f6"],
    # King's Indian: 1. d4 Nf6 2. c4
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -": ["g7g6", "e7e6", "c7c5"],
    # Four Knights Game: 1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq -": ["f1b5", "d2d4", "f1c4", "a2a3", "h2h3"],
    "r1bqkb1r/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq -": ["g8f6", "f8c5", "f8b4"],
    # Queen's Gambit Declined: Classical 3... Be7
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq -": ["c1g5", "c1f4", "c4d5", "e2e3", "f3e5"],
    # Sicilian Classical / Dragon Yugoslav Setup
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ -": ["d4b3", "c3d5", "d1d2", "f1c4"],
    # Carlsbad Structure Mobilization
    "r1b2rk1/pp1nqppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R2Q1RK1 w - -": ["a1d1", "a1c1", "c4d5", "f1e1", "d1e2", "f3e5"],
    "r1b2rk1/ppqn1ppp/2p1pn2/3p4/2PP4/1PNBPN2/P4PPP/R2Q1RK1 w - -": ["b3b4", "a1c1", "d1c2", "f1e1"],
    "r2q1rk1/pp1nbppp/2p1pn2/8/3P4/2NB1N2/PPP2PPP/R1BQ1RK1 w - -": ["c1f4", "c1g5", "d1e2", "f1e1"],
    # Giuoco Pianissimo / Italian Development & Prophylaxis
    "r1bq1rk1/ppp1bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 b - -": ["c8e6", "c8g4", "h7h6", "a7a6", "c6a5"],
    "r1bq1rk1/ppp2ppp/2np4/2b1p3/2B1P1n1/2NP1N2/PPP2PPP/R1BQ1RK1 w - -": ["h2h3", "c1g5", "d1e2", "a2a3"],
    "r1b1k2r/pppp1ppp/2n5/4p3/2B1P1nq/3P1N2/PPP2PPP/RNBQ1RK1 w kq -": ["f3h4", "h2h3", "c4f7", "d1e2"],
    "r1bqk2r/pppp1ppp/2n5/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq -": ["c1d2", "c1e3", "a2a3", "e1g1"],
    "r1bqkb1r/ppp2ppp/2np1n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": ["a7a6", "c8d7", "f8e7", "f6e4"],
    "r1b1kbnr/ppppqppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": ["b1c3", "d2d3", "f1c4", "d2d4", "b1a3"],
}


class MushroomBodyOpeningMemory:
    """Simulates associative opening imprinting in Kenyon Cells -> MBON connections."""

    def __init__(self, memory_strength: float = 350.0):
        self.memory_strength = memory_strength
        self.book = GM_OPENING_BOOK

    def get_memory_bias(self, board: chess.Board, move: chess.Move) -> float:
        """Query associative memory for candidate move.

        Returns associative MBON drive bonus (0.0 to memory_strength).
        """
        # Only active during the opening phase (first 15 moves)
        if len(board.move_stack) > 30:
            return 0.0

        epd = board.epd()
        recommended = self.book.get(epd)

        if recommended and move.uci() in recommended:
            idx = recommended.index(move.uci())
            bonus = self.memory_strength * (1.0 - 0.15 * idx)
            return max(50.0, bonus)

        return 0.0
