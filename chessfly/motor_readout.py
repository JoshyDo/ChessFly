"""Motor Readout & Linear Projection Layer.

Extracts feature states from the ~1,409 descending motor neurons (DNs):
    Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV

Maps motor activity to move preference scores via linear projection matrix W_out:
    Score(m) = W_out^T * F_DN(m) + b
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from chessfly.circuit_registry import CircuitRegistry
from chessfly.simulator import LifSimulator


class MotorReadoutLayer:
    """Linear output projection layer converting 1,409 descending motor neuron states into move scores."""

    def __init__(
        self,
        circuits: CircuitRegistry,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
    ):
        self.circuits = circuits
        self.num_motor_neurons = len(circuits.descending_neurons)
        if self.num_motor_neurons == 0:
            raise ValueError("CircuitRegistry has no descending neurons registered.")

        if weights is not None:
            assert len(weights) == self.num_motor_neurons, (
                f"Weight vector length {len(weights)} must match DN count {self.num_motor_neurons}"
            )
            self.weights = weights.astype(np.float32).copy()
        else:
            self.weights = self._initialize_biological_priors()

        self.bias = float(bias)

    def _initialize_biological_priors(self) -> np.ndarray:
        """Initialize projection weights based on known Drosophila motor pathways.

        - Negative weights on Giant Fiber escape pathway (DNp01, DNp02, DNp06)
        - Positive weights on Forward approach pathway (DNa01, DNa02, DNp09)
        - Small regularized baseline on general descending population
        """
        w = np.zeros(self.num_motor_neurons, dtype=np.float32)
        dn_list = self.circuits.descending_neurons.tolist()
        dn_to_local = {dn_idx: i for i, dn_idx in enumerate(dn_list)}

        # 1. Strong negative valence on Giant Fiber escape neurons:
        # Firing in DNp01/02/06 indicates severe looming danger / blunder
        for idx in self.circuits.escape_readout_neurons:
            if idx in dn_to_local:
                w[dn_to_local[idx]] = -15.0

        # Giant fiber proper (DNp01) gets massive rejection penalty
        for idx in self.circuits.dnp01_giant_fiber:
            if idx in dn_to_local:
                w[dn_to_local[idx]] = -25.0

        # 2. Positive valence on Approach motor neurons:
        # Firing in DNa01/02/DNp09 indicates target pursuit / capture / advance
        for idx in self.circuits.approach_readout_neurons:
            if idx in dn_to_local:
                w[dn_to_local[idx]] = +12.0

        # Approach steering / forward motion
        for idx in self.circuits.dna01:
            if idx in dn_to_local:
                w[dn_to_local[idx]] = +18.0

        # 3. Small positive prior on remaining DNs to encourage active play
        for i in range(self.num_motor_neurons):
            if w[i] == 0.0:
                w[i] = 0.05

        return w

    def extract_motor_features(self, simulator: LifSimulator) -> np.ndarray:
        """Extract motor feature vector: Feature_i = spikes_i + (Vm_i - V_rest) / 7.0 mV."""
        features = simulator.get_motor_features(self.circuits.descending_neurons)
        return features

    def compute_score(self, motor_features: np.ndarray) -> float:
        """Compute move preference score: W_out^T * F_DN + b."""
        return float(np.dot(self.weights, motor_features) + self.bias)

    def train_projection(
        self,
        feature_matrix: np.ndarray,
        target_evaluations: np.ndarray,
        regularization_lambda: float = 1e-2,
    ):
        """Train linear projection weights using L2-regularized Ridge Regression.

        feature_matrix: [N, num_motor_neurons]
        target_evaluations: [N]
        """
        N, D = feature_matrix.shape
        assert D == self.num_motor_neurons

        # Center data
        x_mean = np.mean(feature_matrix, axis=0)
        y_mean = np.mean(target_evaluations)

        X_c = feature_matrix - x_mean
        Y_c = target_evaluations - y_mean

        # Ridge regression: (X^T X + lambda * I)^-1 X^T Y
        reg_matrix = regularization_lambda * np.eye(D, dtype=np.float32)
        xtx = np.dot(X_c.T, X_c) + reg_matrix
        xty = np.dot(X_c.T, Y_c)

        learned_w = np.linalg.solve(xtx, xty)

        # Blend learned weights with biological priors
        self.weights = (0.7 * learned_w + 0.3 * self.weights).astype(np.float32)
        self.bias = float(y_mean - np.dot(self.weights, x_mean))

    def save(self, filepath: Path):
        """Save weights and bias."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(filepath, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, filepath: Path, circuits: CircuitRegistry) -> "MotorReadoutLayer":
        """Load weights and bias from file."""
        data = np.load(filepath)
        return cls(circuits=circuits, weights=data["weights"], bias=float(data["bias"]))
