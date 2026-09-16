"""Unit tests for the Sensory Feature Encoder and 180 Hz Poisson spike trains."""

import chess
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome
from chessfly.chess_encoder import ChessSensoryEncoder


@pytest.fixture
def encoder():
    graph = build_canonical_drosophila_connectome(num_total_neurons=3000, random_seed=42)
    return ChessSensoryEncoder(graph.circuits)


def test_hanging_piece_triggers_looming_threat(encoder):
    """Verify that hanging a major piece triggers strong looming threat (LPLC2/LC4)."""
    # 1. e4 e5 2. Nf3 Nc6 3. Bc4
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")

    # Sound move: 3... Nf6
    t_sound, p_sound, m_sound = encoder.extract_features(board, chess.Move.from_uci("g8f6"))

    # Blunder move: 3... Nd4 4. Bxf7+ or 3... a6? Let's test hanging Queen blunder:
    # 3... d5 (Pawn hangs to Bxd5 or exd5)
    t_pawn_hang, _, _ = encoder.extract_features(board, chess.Move.from_uci("d7d5"))

    # Severe blunder: Black moves Queen into attack of White's knight on f3: 3... Qh4??
    t_blunder, p_blunder, m_blunder = encoder.extract_features(board, chess.Move.from_uci("d8h4"))

    assert t_blunder > 0.8, f"Hanging Queen must produce near-maximum looming threat, got {t_blunder}"
    assert t_blunder > t_sound, "Blunder threat must be strictly higher than sound move"
    assert m_blunder["hanging_material"] >= 9.0, "Hanging material metric must detect Queen"


def test_piece_capture_triggers_target_pursuit(encoder):
    """Verify that capturing an undefended piece excites target pursuit neurons (LC10a/LC9)."""
    # Black has hung an undefended knight on e5
    board = chess.Board("r1bqkbnr/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")

    # Capture knight on e5: 4. Nxe5
    capture_move = chess.Move.from_uci("f3e5")
    t_cap, p_cap, m_cap = encoder.extract_features(board, capture_move)

    # Quiet non-capture move: 4. d3
    quiet_move = chess.Move.from_uci("d2d3")
    t_quiet, p_quiet, _ = encoder.extract_features(board, quiet_move)

    assert p_cap > p_quiet, "Piece capture must produce higher pursuit intensity than quiet move"
    assert m_cap["captured_val"] == 3.0, "Should record captured Knight value"


def test_pawn_promotion_triggers_pursuit(encoder):
    """Verify that pawn promotions strongly excite target pursuit neurons."""
    # White pawn on e7 ready to promote
    board = chess.Board("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")
    promo_move = chess.Move.from_uci("e7e8q")

    t, p, m = encoder.extract_features(board, promo_move)
    assert p >= 0.8, f"Queen promotion must produce high pursuit intensity, got {p}"


def test_180hz_poisson_spike_train_generation(encoder):
    """Verify that Poisson spike trains are generated at up to 180 Hz over 1,000 ms."""
    duration_ms = 1000.0
    dt_ms = 0.2  # 5,000 steps

    events = encoder.generate_poisson_spike_trains(
        threat_intensity=1.0,
        pursuit_intensity=0.0,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        base_rate_hz=180.0,
    )

    # Total spikes in looming neurons over 1.0 second
    total_spikes = sum(len(spikers) for spikers in events.values())
    num_looming_neurons = len(encoder.circuits.looming_threat_inputs)

    # Expected mean spikes per neuron in 1.0 second at 180 Hz = ~180 spikes
    mean_spikes_per_neuron = total_spikes / num_looming_neurons
    print(f"Mean spikes per looming neuron: {mean_spikes_per_neuron:.1f} (expected ~180 Hz)")

    # Poisson rate should be approximately 180 Hz (+- 15%)
    assert 150.0 <= mean_spikes_per_neuron <= 210.0, f"Expected ~180 Hz firing rate, got {mean_spikes_per_neuron}"
