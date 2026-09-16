"""Unit tests for dopaminergic reinforcement learning (PAM11 reward & PPL101 aversive)."""

import chess
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.simulator import LifSimulator
from chessfly.chess_encoder import ChessSensoryEncoder
from chessfly.dopamine_rl import DopamineRLTrainer


@pytest.fixture
def rl_setup():
    graph = build_canonical_drosophila_connectome(num_total_neurons=4000, random_seed=42)
    sim = LifSimulator(graph.indptr, graph.indices, graph.weights, graph.num_neurons, dt_ms=0.5)
    encoder = ChessSensoryEncoder(graph.circuits)
    trainer = DopamineRLTrainer(sim, graph.circuits, encoder, learning_rate_eta=0.01)
    return trainer, sim, graph


def test_dopamine_reward_pam11_trigger(rl_setup):
    """Favorable capture should stimulate PAM11 dopamine reward cells (+1.0)."""
    trainer, _, _ = rl_setup
    # White knight captures hanging black knight on e5
    board = chess.Board("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    capture_move = chess.Move.from_uci("f3e5")

    rf = trainer.compute_reinforcement(board, capture_move)
    assert rf.dan_signal == +1.0, f"Favorable capture must trigger PAM11 reward (+1.0), got {rf.dan_signal}"
    assert rf.is_favorable_capture is True
    assert rf.is_blunder is False


def test_dopamine_aversive_ppl101_trigger(rl_setup):
    """Blunder hanging material should stimulate PPL101 dopamine aversive cells (-1.0)."""
    trainer, _, _ = rl_setup
    # White moves knight into attack of black pawn: 3. Ne5?? hanging to dxe5
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    blunder_move = chess.Move.from_uci("f3e5")

    rf = trainer.compute_reinforcement(board, blunder_move)
    assert rf.dan_signal == -1.0, f"Blunder must trigger PPL101 aversive signal (-1.0), got {rf.dan_signal}"
    assert rf.is_blunder is True


def test_kc_to_mbon_synaptic_weight_modification(rl_setup):
    """Plasticity must modify KC->MBON connection weights in the LIF network."""
    trainer, sim, graph = rl_setup
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    blunder_move = chess.Move.from_uci("f3e5")

    # Initial weights snapshot
    initial_weights = sim.weights.copy()

    # Apply single training trial
    modified_count = trainer.train_single_move(
        board,
        blunder_move,
        simulation_duration_ms=100.0,
        dt_ms=0.5,
    )

    assert modified_count > 0, "Dopaminergic plasticity must update KC->MBON synapses"
    assert trainer.total_plastic_modifications == modified_count
