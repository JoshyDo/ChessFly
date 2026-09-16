"""Sensory Feature Encoder: Maps candidate chess board states to biological visual neurons.

Encodes:
1. Looming Threat & Blunder Defense:
   - Hanging pieces (undefended or underdefended pieces attacked by opponent)
   - Tactical forks against friendly high-value pieces
   - King threats (checks, mating attacks)
   -> Mapped directly to looming projection neurons (LPLC2, LC4)
   -> Triggers downstream Giant Fiber escape pathway (DNp01, DNp02, DNp06)

2. Target Pursuit & Material Capture:
   - Piece captures (clean captures, favorable exchanges)
   - Pawn promotions and advanced passed pawns
   - Central space control (attacks/occupancy of e4, d4, e5, d5, c4, f4, c5, f5)
   -> Mapped directly to small-object pursuit neurons (LC10a, LC9)
   -> Triggers downstream forward/approach motor neurons (DNa01, DNa02, DNp09)

3. 180 Hz Poisson Spike Injection:
   - Generates Poisson spike trains at up to 180 Hz proportional to sensory feature intensities.
"""

from typing import Dict, List, Optional, Tuple
import chess
import numpy as np

from chessfly.circuit_registry import CircuitRegistry

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 100.0,
}

CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]
EXTENDED_CENTER = [chess.C4, chess.D4, chess.E4, chess.F4, chess.C5, chess.D5, chess.E5, chess.F5]


class ChessSensoryEncoder:
    """Extracts biological visual sensory features from chess moves and generates spike trains."""

    def __init__(self, circuits: CircuitRegistry, rng_seed: int = 42):
        self.circuits = circuits
        self.rng = np.random.default_rng(rng_seed)

    def extract_features(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate candidate move 1-ply resulting state.

        Returns:
            threat_intensity (0.0 to 1.0): drives LPLC2/LC4
            pursuit_intensity (0.0 to 1.0): drives LC10a/LC9
            metrics (dict): detailed diagnostic breakdown
        """
        player = board.turn
        opponent = not player

        # Check move legality
        if move not in board.legal_moves:
            # Illegal moves represent catastrophic blunder
            return 1.0, 0.0, {"illegal": 1.0}

        # Check immediate capture metrics from the move itself
        captured_piece = board.piece_at(move.to_square)
        is_en_passant = board.is_en_passant(move)
        captured_val = 0.0
        if captured_piece:
            captured_val = PIECE_VALUES.get(captured_piece.piece_type, 0.0)
        elif is_en_passant:
            captured_val = 1.0

        moving_piece = board.piece_at(move.from_square)
        moving_val = PIECE_VALUES.get(moving_piece.piece_type if moving_piece else chess.PAWN, 1.0)
        is_promotion = move.promotion is not None

        # Simulate 1-ply forward state
        next_board = board.copy()
        next_board.push(move)

        # -------------------------------------------------------------
        # 1. Looming Threat Analysis (on next_board from player's POV)
        # -------------------------------------------------------------
        threat_score = 0.0
        pursuit_score = 0.0

        # A. King Threat: Did move leave king in check or face check?
        # Note: In legal moves, our king cannot be in check on our turn end,
        # but opponent's reply moves can deliver check or mate.
        opponent_can_check = False
        opponent_can_mate = False
        for reply in next_board.legal_moves:
            if next_board.gives_check(reply):
                opponent_can_check = True
                nb2 = next_board.copy()
                nb2.push(reply)
                if nb2.is_checkmate():
                    opponent_can_mate = True
                    break

        if opponent_can_mate:
            threat_score += 1.0  # Immediate impending checkmate
        elif opponent_can_check:
            threat_score += 0.35

        # B. Hanging Pieces (friendly pieces attacked by opponent)
        total_hanging_value = 0.0
        for sq in chess.SQUARES:
            piece = next_board.piece_at(sq)
            if piece and piece.color == player:
                val = PIECE_VALUES.get(piece.piece_type, 0.0)
                if piece.piece_type == chess.KING:
                    continue

                opp_attackers = list(next_board.attackers(opponent, sq))
                if opp_attackers:
                    defenders = list(next_board.attackers(player, sq))
                    if not defenders:
                        # Completely undefended hanging piece!
                        total_hanging_value += val
                    else:
                        # Defended, but attacked by lower-value piece (e.g. pawn attacking Queen)
                        min_attacker_val = min(
                            PIECE_VALUES.get(
                                next_board.piece_at(att).piece_type if next_board.piece_at(att) else chess.PAWN,
                                1.0
                            )
                            for att in opp_attackers
                        )
                        if min_attacker_val < val:
                            total_hanging_value += (val - min_attacker_val)

        # Tactical Fork detection (opponent can attack two major pieces)
        fork_threat = 0.0
        for reply in next_board.legal_moves:
            to_sq = reply.to_square
            # Check what friendly pieces this reply square would attack
            attacked_friendly = [
                next_board.piece_at(sq)
                for sq in chess.SQUARES
                if next_board.piece_at(sq) and next_board.piece_at(sq).color == player
                and next_board.is_pinned(player, sq) is False
                and sq in next_board.attacks(to_sq)
            ]
            high_value_attacked = [
                p for p in attacked_friendly
                if p and PIECE_VALUES.get(p.piece_type, 0.0) >= 3.0
            ]
            if len(high_value_attacked) >= 2:
                fork_threat = max(fork_threat, 0.5)

        # C. Impending attack on our undefended pieces
        impending_threat = 0.0
        for reply in next_board.legal_moves:
            to_sq = reply.to_square
            for sq in next_board.attacks(to_sq):
                p = next_board.piece_at(sq)
                if p and p.color == player and p.piece_type != chess.KING:
                    if not next_board.is_attacked_by(player, sq):
                        val = PIECE_VALUES.get(p.piece_type, 1.0)
                        if val >= 3.0:
                            impending_threat = max(impending_threat, val / 12.0)

        # King safety & opening development
        if moving_piece and moving_piece.piece_type == chess.KING:
            if board.is_castling(move):
                pursuit_score += 0.45  # Castling is highly beneficial
            elif not board.is_check() and len(board.move_stack) < 25:
                threat_score += 0.35   # Unnecessary king moves lose castling and expose king

        # Normalize threat intensity into [0.0, 1.0]
        threat_score += (total_hanging_value / 3.0) + fork_threat + impending_threat
        threat_intensity = float(np.clip(threat_score, 0.0, 1.0))

        # -------------------------------------------------------------
        # 2. Target Pursuit Analysis (favorable action, approach)
        # -------------------------------------------------------------
        # A. Captures and favorable exchanges
        if captured_val > 0:
            # Check if target square is defended by opponent
            opp_defenders = list(next_board.attackers(opponent, move.to_square))
            if not opp_defenders:
                # Free capture!
                pursuit_score += min(1.0, captured_val / 3.0 + 0.4)
            else:
                # Exchange: net trade
                net_trade = captured_val - moving_val
                if net_trade >= 0:
                    pursuit_score += (net_trade + 1.0) / 4.0
                else:
                    # Negative trade: small capture credit, but threat will penalize
                    pursuit_score += 0.05

        # B. Promotions and passed pawns
        if is_promotion:
            promo_val = PIECE_VALUES.get(move.promotion, 9.0)
            pursuit_score += (promo_val - 1.0) / 8.0
        elif moving_piece and moving_piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            if player == chess.WHITE and rank >= 5:
                pursuit_score += (rank - 4) * 0.15
            elif player == chess.BLACK and rank <= 2:
                pursuit_score += (3 - rank) * 0.15

        # C. Central Space Control & Piece Development
        center_control = 0
        for sq in CENTER_SQUARES:
            p = next_board.piece_at(sq)
            if p and p.color == player:
                center_control += 2
            if next_board.is_attacked_by(player, sq):
                center_control += 1

        for sq in EXTENDED_CENTER:
            if next_board.is_attacked_by(player, sq):
                center_control += 0.5

        pursuit_score += (center_control / 16.0) * 0.35

        # D. Resolving or giving check
        if board.is_check() and not next_board.is_check():
            # Interposing or developing block to resolve check
            if moving_piece and moving_piece.piece_type != chess.KING:
                pursuit_score += 0.35
        if next_board.is_check():
            pursuit_score += 0.25

        # E. Piece Development (developing knights and bishops off back rank)
        if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if (player == chess.WHITE and from_rank == 0 and to_rank in [1, 2, 3]) or \
               (player == chess.BLACK and from_rank == 7 and to_rank in [4, 5, 6]):
                pursuit_score += 0.20

        pursuit_intensity = float(np.clip(pursuit_score, 0.0, 1.0))

        metrics = {
            "threat_intensity": threat_intensity,
            "pursuit_intensity": pursuit_intensity,
            "hanging_material": total_hanging_value,
            "captured_val": captured_val,
            "fork_threat": fork_threat,
            "opponent_can_check": float(opponent_can_check),
            "opponent_can_mate": float(opponent_can_mate),
        }
        return threat_intensity, pursuit_intensity, metrics

    def generate_poisson_spike_trains(
        self,
        threat_intensity: float,
        pursuit_intensity: float,
        duration_ms: float = 1000.0,
        dt_ms: float = 0.2,
        base_rate_hz: float = 180.0,
    ) -> Dict[int, List[int]]:
        """Generate 180 Hz Poisson-style spike trains for visual neurons over 1,000 ms.

        Returns:
            sparse event map: {step_index: [target_neuron_indices]}
        """
        num_steps = int(round(duration_ms / dt_ms))
        dt_sec = dt_ms / 1000.0

        looming_neurons = self.circuits.looming_threat_inputs
        pursuit_neurons = self.circuits.target_pursuit_inputs

        events: Dict[int, List[int]] = {}

        # Threat spikes -> Looming neurons (LPLC2, LC4)
        if threat_intensity > 0.01 and len(looming_neurons) > 0:
            rate_threat = base_rate_hz * threat_intensity
            p_threat = 1.0 - np.exp(-rate_threat * dt_sec)
            # Sample spikes per step for looming neurons
            spike_mask = self.rng.random((num_steps, len(looming_neurons))) < p_threat
            for step_idx in range(num_steps):
                spikers = looming_neurons[spike_mask[step_idx]]
                if len(spikers) > 0:
                    events.setdefault(step_idx, []).extend(spikers.tolist())

        # Pursuit spikes -> Target pursuit neurons (LC10a, LC9)
        if pursuit_intensity > 0.01 and len(pursuit_neurons) > 0:
            rate_pursuit = base_rate_hz * pursuit_intensity
            p_pursuit = 1.0 - np.exp(-rate_pursuit * dt_sec)
            spike_mask = self.rng.random((num_steps, len(pursuit_neurons))) < p_pursuit
            for step_idx in range(num_steps):
                spikers = pursuit_neurons[spike_mask[step_idx]]
                if len(spikers) > 0:
                    events.setdefault(step_idx, []).extend(spikers.tolist())

        return events
