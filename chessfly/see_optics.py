"""Static Exchange Evaluation (SEE) Optics (ChessFly V3).

Biophysical analogy: Predatory visual looming optics.
Calculates net material balance of exchanging pieces on a target square without tree search.
Orders attackers by value: Pawn (100) < Knight (320) <= Bishop (330) < Rook (500) < Queen (900) < King (20000).

Net material payoff:
- Negative trade -> Injects looming threat to LPLC2 / LC4 ommatidial columns.
- Positive trade -> Injects target pursuit to LC10a / LC9 ommatidial columns.
"""

from typing import Dict, List, Optional, Tuple
import chess

PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def see_square(board: chess.Board, square: chess.Square) -> int:
    """Calculates static exchange evaluation (SEE) score in centipawns on square.

    Returns net material balance for the side to move capturing on square.
    """
    target = board.piece_at(square)
    if not target:
        return 0

    capture_val = [PIECE_VALUES.get(target.piece_type, 100)]
    occupied = board.occupied
    turn = board.turn

    while True:
        attackers = []
        for s in board.attackers(turn, square):
            if (chess.BB_SQUARES[s] & occupied):
                p = board.piece_at(s)
                pv = PIECE_VALUES.get(p.piece_type, 100) if p else 100
                attackers.append((pv, s))
        if not attackers:
            break

        attackers.sort(key=lambda x: x[0])
        pv, att_sq = attackers[0]
        occupied &= ~chess.BB_SQUARES[att_sq]
        capture_val.append(pv)
        turn = not turn

    n = len(capture_val) - 1
    balance = 0
    for k in range(n - 1, -1, -1):
        balance = max(0, capture_val[k] - balance)

    return balance


static_exchange_evaluation = see_square


def evaluate_move_see(board: chess.Board, move: chess.Move) -> int:
    """Evaluate SEE balance for candidate move in centipawns.

    Returns:
        Positive centipawns: net material gain.
        Negative centipawns: net material blunder.
    """
    target_piece = board.piece_at(move.to_square)
    captured_val = (
        PIECE_VALUES.get(target_piece.piece_type, 100)
        if target_piece
        else (100 if board.is_en_passant(move) else 0)
    )

    next_b = board.copy()
    next_b.push(move)

    opp = next_b.turn
    opp_attackers = list(next_b.attackers(opp, move.to_square))
    if not opp_attackers:
        return captured_val

    see_loss = see_square(next_b, move.to_square)
    return captured_val - see_loss
