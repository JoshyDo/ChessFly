"""Unit tests for Connectome Ingestion and CSR FP16 compression."""

from pathlib import Path
import numpy as np
import pytest

from chessfly.connectome_loader import build_canonical_drosophila_connectome, ConnectomeGraph


def test_canonical_drosophila_connectome_structure():
    """Verify that canonical Drosophila connectome has all required biological circuits."""
    graph = build_canonical_drosophila_connectome(num_total_neurons=8000, random_seed=42)

    # 1. Check basic graph properties
    assert graph.num_neurons == 8000
    assert len(graph.indptr) == 8001
    assert len(graph.indices) == len(graph.weights)
    assert len(graph.indices) > 50000

    # 2. Check visual looming threat neurons (LPLC2 + LC4)
    looming = graph.circuits.looming_threat_inputs
    assert len(looming) > 0
    assert len(graph.circuits.lplc2) == 120
    assert len(graph.circuits.lc4) == 100

    # 3. Check target pursuit neurons (LC10a + LC9)
    pursuit = graph.circuits.target_pursuit_inputs
    assert len(pursuit) > 0
    assert len(graph.circuits.lc10a) == 140
    assert len(graph.circuits.lc9) == 100

    # 4. Check Giant Fiber escape pathway
    assert len(graph.circuits.dnp01_giant_fiber) == 2
    assert len(graph.circuits.escape_readout_neurons) >= 10

    # 5. Check Approach motor pathway
    assert len(graph.circuits.dna01) == 2
    assert len(graph.circuits.dna02) == 2
    assert len(graph.circuits.approach_readout_neurons) >= 8

    # 6. Check Descending motor neurons: must match Drosophila anatomy (~1,409 DNs)
    assert len(graph.circuits.descending_neurons) == 1409

    # 7. Check Mushroom body & dopamine cells
    assert len(graph.circuits.kenyon_cells) == 2000
    assert len(graph.circuits.mbon_cells) >= 40
    assert len(graph.circuits.pam11_reward) > 0
    assert len(graph.circuits.ppl101_aversive) > 0


def test_csr_fp16_serialization(tmp_path):
    """Verify that connectome compresses into Sparse CSR format with FP16 weights."""
    graph = build_canonical_drosophila_connectome(num_total_neurons=4000, random_seed=123)
    save_file = tmp_path / "test_graph.npz"

    graph.save_compressed(save_file)
    assert save_file.exists()

    # Load file and check weights dtype is FP16 on disk
    raw_npz = np.load(save_file)
    assert raw_npz["weights"].dtype == np.float16

    # Load through ConnectomeGraph API
    loaded = ConnectomeGraph.load_compressed(save_file)
    assert loaded.num_neurons == 4000
    assert len(loaded.indices) == len(graph.indices)
    assert len(loaded.circuits.descending_neurons) == 1409
    # Loaded weights should be float32 for high-precision simulation
    assert loaded.weights.dtype == np.float32
