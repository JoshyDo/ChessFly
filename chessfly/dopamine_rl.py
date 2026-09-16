"""Dopaminergic Reinforcement Learning & Synaptic Plasticity.

Implements Drosophila Mushroom Body 3-factor learning rule:
- Reward signals (favorable trades, checks, winning material) stimulate PAM11 dopamine reward cells (D = +1.0)
- Aversive signals (losing material, hanging pieces, blunders) stimulate PPL101 dopamine cells (D = -1.0)
- Plasticity operates across Kenyon Cells (KCs) -> Mushroom Body Output Neurons (MBONs)

Provides high-volume vectorized training framework to optimize connectome weights.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np

from chessfly.circuit_registry import CircuitRegistry
from chessfly.simulator import LifSimulator
from chessfly.chess_encoder import ChessSensoryEncoder, PIECE_VALUES


@dataclass
class ReinforcementSignal:
    """Calculated dopamine signal for a given candidate move."""
    dan_signal: float  # +1.0 for PAM11 reward, -1.0 for PPL101 aversive, 0.0 neutral
    reward_magnitude: float
    is_blunder: bool
    is_favorable_capture: bool
    is_check: bool
    material_delta: float


class DopamineRLTrainer:
    """Manages dopamine-driven credit assignment and synaptic plasticity training."""

    def __init__(
        self,
        simulator: LifSimulator,
        circuits: CircuitRegistry,
        encoder: ChessSensoryEncoder,
        learning_rate_eta: float = 0.002,
    ):
        self.sim = simulator
        self.circuits = circuits
        self.encoder = encoder
        self.eta = learning_rate_eta
        self.total_training_steps = 0
        self.total_plastic_modifications = 0

    def compute_reinforcement(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> ReinforcementSignal:
        """Evaluate reinforcement signal: PAM11 reward vs PPL101 aversive."""
        threat, pursuit, metrics = self.encoder.extract_features(board, move)

        captured_val = metrics.get("captured_val", 0.0)
        hanging_val = metrics.get("hanging_material", 0.0)
        fork_threat = metrics.get("fork_threat", 0.0)

        # Calculate net tactical swing
        net_material = captured_val - hanging_val

        # Aversive trigger: Hanging pieces or fork blunder -> PPL101 stimulation
        is_blunder = (hanging_val >= 1.0) or (fork_threat >= 0.5) or (metrics.get("opponent_can_mate", 0.0) > 0.5)
        # Reward trigger: Favorable trade / capture without hanging pieces
        is_favorable = (captured_val > 0 and hanging_val == 0) or (net_material > 0)

        next_b = board.copy()
        next_b.push(move)
        is_check = next_b.is_check()

        if is_blunder:
            # PPL101 stimulation (negative dopamine signal)
            dan_signal = -1.0
            magnitude = min(1.0, (hanging_val + 1.0) / 5.0)
        elif is_favorable or (is_check and hanging_val == 0):
            # PAM11 stimulation (positive dopamine signal)
            dan_signal = 1.0
            magnitude = min(1.0, (captured_val + 0.5) / 5.0)
        else:
            dan_signal = 0.0
            magnitude = 0.0

        return ReinforcementSignal(
            dan_signal=dan_signal,
            reward_magnitude=magnitude,
            is_blunder=is_blunder,
            is_favorable_capture=is_favorable,
            is_check=is_check,
            material_delta=net_material,
        )

    def train_single_move(
        self,
        board: chess.Board,
        move: chess.Move,
        simulation_duration_ms: float = 500.0,
        dt_ms: float = 0.5,
    ) -> int:
        """Simulate move, compute dopamine reinforcement, and apply KC->MBON plasticity."""
        rf = self.compute_reinforcement(board, move)
        if abs(rf.dan_signal) < 0.01:
            return 0

        # Extract features and generate Poisson spike trains
        threat, pursuit, _ = self.encoder.extract_features(board, move)
        events = self.encoder.generate_poisson_spike_trains(
            threat_intensity=threat,
            pursuit_intensity=pursuit,
            duration_ms=simulation_duration_ms,
            dt_ms=dt_ms,
        )

        # Reset simulator state from rest
        self.sim.reset()

        # Stimulate dopamine cells according to reinforcement:
        # PAM11 cells for positive reward, PPL101 cells for aversive blunder
        if rf.dan_signal > 0 and len(self.circuits.pam11_reward) > 0:
            self.sim.set_sensory_drive(
                self.circuits.pam11_reward,
                np.full(len(self.circuits.pam11_reward), 15.0 * rf.reward_magnitude, dtype=np.float32)
            )
        elif rf.dan_signal < 0 and len(self.circuits.ppl101_aversive) > 0:
            self.sim.set_sensory_drive(
                self.circuits.ppl101_aversive,
                np.full(len(self.circuits.ppl101_aversive), 15.0 * rf.reward_magnitude, dtype=np.float32)
            )

        # Run biophysical simulation window
        self.sim.simulate_window(
            duration_ms=simulation_duration_ms,
            dt_ms=dt_ms,
            sensory_spike_events=events,
        )

        # Apply dopaminergic plasticity to KC -> MBON synapses
        effective_eta = self.eta * rf.reward_magnitude
        modified = self.sim.apply_plasticity(
            kc_indices=self.circuits.kenyon_cells,
            mbon_indices=self.circuits.mbon_cells,
            dan_signal=rf.dan_signal,
            eta=effective_eta,
        )

        self.total_training_steps += 1
        self.total_plastic_modifications += modified
        return modified

    def train_batch(
        self,
        positions: List[Tuple[chess.Board, chess.Move]],
        simulation_duration_ms: float = 200.0,
        dt_ms: float = 0.5,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run vectorized batch training over multiple board positions."""
        total_mod = 0
        blunder_count = 0
        favorable_count = 0

        for idx, (board, move) in enumerate(positions):
            rf = self.compute_reinforcement(board, move)
            if rf.is_blunder:
                blunder_count += 1
            elif rf.is_favorable_capture:
                favorable_count += 1

            mod = self.train_single_move(
                board,
                move,
                simulation_duration_ms=simulation_duration_ms,
                dt_ms=dt_ms,
            )
            total_mod += mod

            if verbose and (idx + 1) % 50 == 0:
                print(f"Trained {idx + 1}/{len(positions)} moves | Plastic updates: {total_mod}")

        return {
            "trained_moves": float(len(positions)),
            "total_modifications": float(total_mod),
            "blunders_encountered": float(blunder_count),
            "favorable_encountered": float(favorable_count),
        }
