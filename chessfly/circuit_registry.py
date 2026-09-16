"""Biological circuit definitions and cell type mappings for Drosophila melanogaster connectome.

Identifies key functional circuits:
- Visual Looming Threat: LPLC2, LC4
- Visual Target Pursuit: LC10a, LC9
- Giant Fiber Escape Pathway: DNp01 (GF), DNp02, DNp06
- Forward/Approach Motor Pathway: DNa01, DNa02, DNp09
- Mushroom Body Memory System: KCs, MBONs, PAM11 (reward), PPL101 (aversive)
- Descending Motor Neurons (~1,409 DNs)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set
import numpy as np


# Exact cell type designations from FlyWire / MaleCNS
LOOMING_THREAT_TYPES = ["LPLC2", "LC4"]
TARGET_PURSUIT_TYPES = ["LC10a", "LC9"]
GIANT_FIBER_ESCAPE_TYPES = ["DNp01", "DNp02", "DNp06"]
FORWARD_APPROACH_TYPES = ["DNa01", "DNa02", "DNp09"]

DOPAMINE_REWARD_TYPES = ["PAM11", "PAM-g5", "PAM01"]
DOPAMINE_AVERSIVE_TYPES = ["PPL101", "PPL1-g1pedc", "PPL102"]

MBON_AVOIDANCE_TYPES = ["MBON01", "MBON03"]
MBON_APPROACH_TYPES = ["MBON11", "MBON-g1pedc", "MBON14"]


@dataclass
class CircuitRegistry:
    """Stores neuron indices for specific Drosophila functional circuits."""
    num_neurons: int = 0
    # Visual input neurons
    lplc2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    lc4: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    lc10a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    lc9: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))

    # Descending motor escape vs approach pathways
    dnp01_giant_fiber: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    dnp02: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    dnp06: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    dna01: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    dna02: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    dnp09: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))

    # All descending motor neurons (~1,409)
    descending_neurons: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))

    # Mushroom body & dopaminergic system
    kenyon_cells: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    mbon_cells: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    pam11_reward: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))
    ppl101_aversive: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint32))

    @property
    def looming_threat_inputs(self) -> np.ndarray:
        """Combined looming threat visual input neurons (LPLC2 + LC4)."""
        return np.unique(np.concatenate([self.lplc2, self.lc4])).astype(np.uint32)

    @property
    def target_pursuit_inputs(self) -> np.ndarray:
        """Combined target pursuit visual input neurons (LC10a + LC9)."""
        return np.unique(np.concatenate([self.lc10a, self.lc9])).astype(np.uint32)

    @property
    def escape_readout_neurons(self) -> np.ndarray:
        """Combined Giant Fiber escape pathway motor neurons (DNp01, DNp02, DNp06)."""
        return np.unique(np.concatenate([self.dnp01_giant_fiber, self.dnp02, self.dnp06])).astype(np.uint32)

    @property
    def approach_readout_neurons(self) -> np.ndarray:
        """Combined forward approach motor neurons (DNa01, DNa02, DNp09)."""
        return np.unique(np.concatenate([self.dna01, self.dna02, self.dnp09])).astype(np.uint32)
