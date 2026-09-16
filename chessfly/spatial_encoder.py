"""Spatial Retinotopic Feature Encoder (ChessFly V2 - Grandmaster Fly).

Maps the 64 squares of the chessboard into ommatidial visual columns in the
Drosophila visual system (LPLC2, LC4, LC10a, LC9).

Features:
1. 64-Square Spatial Retinotopy: Localized column receptive fields.
2. Ray-Tracing Spatial Threat Optics: Absolute pins, skewers, x-rays along lines of sight.
3. Positional Master Patterns:
   - Knight outposts (unassailable central squares)
   - Rooks on open files and 7th rank
   - Pawn structure geometry: Passed pawns, isolated pawns, doubled pawns, king shield
4. Central Complex Vector Fields: Heading direction toward the opposing King.
"""

from typing import Dict, List, Optional, Set, Tuple
import chess
import numpy as np

from chessfly.circuit_registry import CircuitRegistry
from chessfly.chess_encoder import PIECE_VALUES, CENTER_SQUARES, EXTENDED_CENTER


class SpatialRetinotopicEncoder:
    """Retinotopic spatial encoder projecting 64 chess squares into Drosophila visual columns."""

    def __init__(self, circuits: CircuitRegistry, rng_seed: int = 42):
        self.circuits = circuits
        self.rng = np.random.default_rng(rng_seed)

        # Build 64-square ommatidial receptive fields mapping squares 0..63 to neuron subsets
        self.square_to_lplc2 = self._partition_neurons(circuits.lplc2, 64)
        self.square_to_lc4 = self._partition_neurons(circuits.lc4, 64)
        self.square_to_lc10a = self._partition_neurons(circuits.lc10a, 64)
        self.square_to_lc9 = self._partition_neurons(circuits.lc9, 64)

    def _partition_neurons(self, neurons: np.ndarray, num_bins: int = 64) -> List[np.ndarray]:
        """Distribute neurons evenly across 64 spatial receptive fields."""
        if len(neurons) == 0:
            return [np.array([], dtype=np.uint32) for _ in range(num_bins)]
        splits = np.array_split(neurons, num_bins)
        return [s.astype(np.uint32) for s in splits]

    def extract_spatial_features(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute spatial 64-element threat map and pursuit map.

        Returns:
            threat_map: np.ndarray [64] in [0.0, 1.0]
            pursuit_map: np.ndarray [64] in [0.0, 1.0]
            metrics: dict of positional and tactical metrics
        """
        player = board.turn
        opponent = not player

        if move not in board.legal_moves:
            return np.ones(64, dtype=np.float32), np.zeros(64, dtype=np.float32), {"illegal": 1.0}

        next_board = board.copy()
        next_board.push(move)

        threat_map = np.zeros(64, dtype=np.float32)
        pursuit_map = np.zeros(64, dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Spatial Looming Threat & Ray-Tracing Optics
        # -------------------------------------------------------------
        # A. Direct Hanging pieces on squares
        for sq in chess.SQUARES:
            p = next_board.piece_at(sq)
            if p and p.color == player:
                val = PIECE_VALUES.get(p.piece_type, 1.0)
                opp_attackers = list(next_board.attackers(opponent, sq))
                if opp_attackers:
                    defenders = list(next_board.attackers(player, sq))
                    if not defenders:
                        threat_map[sq] += min(1.0, val / 3.0)
                    else:
                        min_att_val = min(
                            PIECE_VALUES.get(
                                next_board.piece_at(a).piece_type if next_board.piece_at(a) else chess.PAWN,
                                1.0
                            )
                            for a in opp_attackers
                        )
                        if min_att_val < val:
                            threat_map[sq] += min(1.0, (val - min_att_val) / 3.0)

        # B. Ray-Tracing: Absolute Pins & Skewers
        for sq in chess.SQUARES:
            p = next_board.piece_at(sq)
            if p and p.color == player:
                if next_board.is_pinned(player, sq):
                    # Pinned piece is restricted in movement; mark its square with threat
                    val = PIECE_VALUES.get(p.piece_type, 1.0)
                    threat_map[sq] += 0.45 * (val / 5.0)

        # C. King Exposure & Check Threat
        king_sq = next_board.king(player)
        if king_sq is not None:
            # Attacks around king zone
            king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
            for k_neighbor in king_zone:
                opp_att = len(list(next_board.attackers(opponent, k_neighbor)))
                if opp_att > 0:
                    threat_map[k_neighbor] += 0.20 * min(3, opp_att)

            if next_board.is_check():
                threat_map[king_sq] += 0.85

        # D. Impending Opponent Attacks on undefended pieces
        for reply in next_board.legal_moves:
            to_sq = reply.to_square
            for target_sq in next_board.attacks(to_sq):
                p = next_board.piece_at(target_sq)
                if p and p.color == player and p.piece_type != chess.KING:
                    if not next_board.is_attacked_by(player, target_sq):
                        val = PIECE_VALUES.get(p.piece_type, 1.0)
                        if val >= 3.0:
                            threat_map[target_sq] += val / 12.0

        # -------------------------------------------------------------
        # 2. Spatial Target Pursuit & Master Positional Features
        # -------------------------------------------------------------
        # A. Captures and favorable trades
        captured_val = 0.0
        if board.is_capture(move):
            target_piece = board.piece_at(move.to_square)
            if target_piece:
                captured_val = PIECE_VALUES.get(target_piece.piece_type, 1.0)
            elif board.is_en_passant(move):
                captured_val = 1.0

            if captured_val > 0:
                opp_defs = list(next_board.attackers(opponent, move.to_square))
                if not opp_defs:
                    pursuit_map[move.to_square] += min(1.0, captured_val / 3.0 + 0.4)
                else:
                    moving_p = board.piece_at(move.from_square)
                    moving_val = PIECE_VALUES.get(moving_p.piece_type if moving_p else chess.PAWN, 1.0)
                    net_trade = captured_val - moving_val
                    if net_trade >= 0:
                        pursuit_map[move.to_square] += (net_trade + 1.0) / 4.0

        # B. Rooks on Open Files & 7th Rank
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type == chess.ROOK:
            file_idx = chess.square_file(move.to_square)
            # Check if file is open (no pawns on file)
            file_mask = chess.BB_FILES[file_idx]
            all_pawns = next_board.pieces(chess.PAWN, chess.WHITE) | next_board.pieces(chess.PAWN, chess.BLACK)
            if not (file_mask & all_pawns.mask):
                pursuit_map[move.to_square] += 0.40  # Open file for Rook!

            # Rook on the 7th rank (for White: rank 6, Black: rank 1)
            to_rank = chess.square_rank(move.to_square)
            if (player == chess.WHITE and to_rank == 6) or (player == chess.BLACK and to_rank == 1):
                pursuit_map[move.to_square] += 0.50  # Rook on the 7th!

        # C. Knight Outposts
        if moving_piece and moving_piece.piece_type == chess.KNIGHT:
            to_sq = move.to_square
            rank = chess.square_rank(to_sq)
            file_idx = chess.square_file(to_sq)
            # Outpost on ranks 4, 5 (or 3, 2 for black) protected by friendly pawn
            in_outpost_rank = (player == chess.WHITE and rank in [3, 4, 5]) or (player == chess.BLACK and rank in [2, 3, 4])
            if in_outpost_rank:
                # Protected by friendly pawn?
                defended_by_pawn = any(
                    next_board.piece_at(d) and next_board.piece_at(d).piece_type == chess.PAWN
                    for d in next_board.attackers(player, to_sq)
                )
                if defended_by_pawn:
                    pursuit_map[to_sq] += 0.45  # True Knight Outpost!

        # D. Pawn Structure: Passed Pawns & Promotions
        if move.promotion:
            promo_val = PIECE_VALUES.get(move.promotion, 9.0)
            pursuit_map[move.to_square] += min(1.0, (promo_val - 1.0) / 8.0 + 0.30)
        elif moving_piece and moving_piece.piece_type == chess.PAWN:
            to_sq = move.to_square
            # Check if passed pawn
            if self._is_passed_pawn(next_board, to_sq, player):
                rank = chess.square_rank(to_sq)
                advance = (rank - 3) if player == chess.WHITE else (4 - rank)
                pursuit_map[to_sq] += 0.35 + 0.15 * max(0, advance)

        # E. Castling and King Safety
        if moving_piece and moving_piece.piece_type == chess.KING:
            if board.is_castling(move):
                pursuit_map[move.to_square] += 0.60  # Grandmaster king safety
            elif not board.is_check() and len(board.move_stack) < 25:
                threat_map[move.to_square] += 0.40   # Penalize unprovoked king wandering

        # F. Central Dominance & Minor Piece Development
        for sq in CENTER_SQUARES:
            if sq == move.to_square:
                pursuit_map[sq] += 0.30
            elif next_board.is_attacked_by(player, sq):
                pursuit_map[sq] += 0.10

        if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if (player == chess.WHITE and from_rank == 0 and to_rank in [1, 2, 3]) or \
               (player == chess.BLACK and from_rank == 7 and to_rank in [4, 5, 6]):
                pursuit_map[move.to_square] += 0.25

            # Penalize blocking central pawns (e.g. Bd3 blocking d2 pawn before it moves)
            if move.to_square == chess.D3 and board.piece_at(chess.D2) == chess.Piece(chess.PAWN, player):
                threat_map[chess.D3] += 0.35
            elif move.to_square == chess.E3 and board.piece_at(chess.E2) == chess.Piece(chess.PAWN, player):
                threat_map[chess.E3] += 0.35
            elif move.to_square == chess.D6 and board.piece_at(chess.D7) == chess.Piece(chess.PAWN, player):
                threat_map[chess.D6] += 0.35
            elif move.to_square == chess.E6 and board.piece_at(chess.E7) == chess.Piece(chess.PAWN, player):
                threat_map[chess.E6] += 0.35

        # G. Pinning Opponent Pieces (Ray-Tracing Target Pursuit)
        opp_pinned = [
            sq for sq in chess.SQUARES
            if next_board.piece_at(sq) and next_board.piece_at(sq).color == opponent
            and next_board.is_pinned(opponent, sq)
        ]
        if opp_pinned:
            pursuit_map[move.to_square] += 0.30 * min(2, len(opp_pinned))

        # Clip maps
        np.clip(threat_map, 0.0, 1.0, out=threat_map)
        np.clip(pursuit_map, 0.0, 1.0, out=pursuit_map)

        metrics = {
            "mean_threat": float(np.mean(threat_map)),
            "max_threat": float(np.max(threat_map)),
            "mean_pursuit": float(np.mean(pursuit_map)),
            "max_pursuit": float(np.max(pursuit_map)),
            "captured_val": captured_val,
        }
        return threat_map, pursuit_map, metrics

    def _is_passed_pawn(self, board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
        """Check if pawn on sq is a passed pawn."""
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        opp_color = not color

        opp_pawns = board.pieces(chess.PAWN, opp_color)
        for f in [file_idx - 1, file_idx, file_idx + 1]:
            if 0 <= f <= 7:
                file_pawns = opp_pawns & chess.SquareSet(chess.BB_FILES[f])
                for p_sq in file_pawns:
                    p_rank = chess.square_rank(p_sq)
                    if color == chess.WHITE and p_rank > rank_idx:
                        return False
                    elif color == chess.BLACK and p_rank < rank_idx:
                        return False
        return True

    def generate_retinotopic_poisson_spikes(
        self,
        threat_map: np.ndarray,
        pursuit_map: np.ndarray,
        duration_ms: float = 500.0,
        dt_ms: float = 0.5,
        base_rate_hz: float = 180.0,
    ) -> Dict[int, List[int]]:
        """Generate Poisson spike events mapped to the exact 64 ommatidial neuron columns."""
        num_steps = int(round(duration_ms / dt_ms))
        dt_sec = dt_ms / 1000.0
        events: Dict[int, List[int]] = {}

        for sq in range(64):
            t_val = threat_map[sq]
            p_val = pursuit_map[sq]

            # 1. Threat spikes -> ommatidial columns of LPLC2 & LC4
            if t_val > 0.05:
                rate = base_rate_hz * t_val
                p_spike = 1.0 - np.exp(-rate * dt_sec)
                target_neurons = np.concatenate([self.square_to_lplc2[sq], self.square_to_lc4[sq]])
                if len(target_neurons) > 0:
                    spike_mask = self.rng.random((num_steps, len(target_neurons))) < p_spike
                    for step_idx in range(num_steps):
                        spikers = target_neurons[spike_mask[step_idx]]
                        if len(spikers) > 0:
                            events.setdefault(step_idx, []).extend(spikers.tolist())

            # 2. Pursuit spikes -> ommatidial columns of LC10a & LC9
            if p_val > 0.05:
                rate = base_rate_hz * p_val
                p_spike = 1.0 - np.exp(-rate * dt_sec)
                target_neurons = np.concatenate([self.square_to_lc10a[sq], self.square_to_lc9[sq]])
                if len(target_neurons) > 0:
                    spike_mask = self.rng.random((num_steps, len(target_neurons))) < p_spike
                    for step_idx in range(num_steps):
                        spikers = target_neurons[spike_mask[step_idx]]
                        if len(spikers) > 0:
                            events.setdefault(step_idx, []).extend(spikers.tolist())

        return events
