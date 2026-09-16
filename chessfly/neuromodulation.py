"""Neuromodulation Controller (ChessFly V2).

Models biophysical Drosophila neuromodulators:
1. Octopamine (OA): Fight-or-flight, aggression, and tactical initiative.
   - High attack tension on opposing King triggers OA release.
   - Lowers action potential threshold (V_thresh - 3.0 mV) and boosts synaptic gain.
2. Serotonin (5-HT): Risk aversion and defensive stability.
   - Modulates steady-state membrane conductances under defensive pressure.
3. Dopamine (DA): Reinforcement credit assignment (PAM11 / PPL101).
"""

from dataclasses import dataclass
import chess
import numpy as np


@dataclass
class NeuromodulatoryState:
    """Biophysical neuromodulator concentrations."""
    octopamine: float      # 0.0 to 1.0 (aggression/tactical focus)
    serotonin: float       # 0.0 to 1.0 (patience/defensive stability)
    conductance_gain: float # 1.0 to 2.0 (synaptic amplification)


class NeuromodulationController:
    """Calculates chemical neuromodulator levels based on board tension."""

    def compute_state(self, board: chess.Board, move: chess.Move) -> NeuromodulatoryState:
        player = board.turn
        opponent = not player

        next_b = board.copy()
        next_b.push(move)

        # 1. Octopamine (OA) - Attack Intensity & Tactical Initiative
        # Triggered by: giving check, attacking opponent king zone, multiple piece attacks
        oa_score = 0.0

        if next_b.is_check():
            oa_score += 0.50

        opp_king_sq = next_b.king(opponent)
        if opp_king_sq is not None:
            king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[opp_king_sq])
            friendly_attacks_on_king = sum(len(list(next_b.attackers(player, sq))) for sq in king_zone)
            oa_score += 0.10 * min(5, friendly_attacks_on_king)

        if board.is_capture(move):
            oa_score += 0.25

        octopamine = float(np.clip(oa_score, 0.0, 1.0))

        # 2. Serotonin (5-HT) - Defensive vigilance
        # Triggered by: our king under pressure, opponent active pieces
        ht_score = 0.0
        friendly_king_sq = next_b.king(player)
        if friendly_king_sq is not None:
            king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[friendly_king_sq])
            enemy_attacks_on_king = sum(len(list(next_b.attackers(opponent, sq))) for sq in king_zone)
            ht_score += 0.15 * min(4, enemy_attacks_on_king)

        serotonin = float(np.clip(ht_score, 0.0, 1.0))

        # 3. Synaptic Conductance Gain
        # Baseline 1.0, boosted up to 1.5 during sharp octopaminergic attacks
        conductance_gain = float(1.0 + 0.5 * octopamine)

        return NeuromodulatoryState(
            octopamine=octopamine,
            serotonin=serotonin,
            conductance_gain=conductance_gain,
        )
