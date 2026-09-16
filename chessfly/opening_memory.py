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
