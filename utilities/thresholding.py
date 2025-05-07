import numpy as np
from typing import Literal, Optional
from scipy.stats import genpareto, genextreme


class ThresholdingDetector:
    """
    A class for thresholding anomaly scores using static, quantile, POT, or EVT-based methods.
    """

    def __init__(
        self,
        method: Literal["static", "pot", "quantile", "evt"] = "static",
        k: float = 3.0,
        quantile: float = 0.99,
        pot_q: float = 0.98,
        pot_level: float = 0.01,
        evt_block_size: int = 50,
        evt_fpr: float = 0.01,
    ):
        """
        Initializes the thresholding detector.

        Args:
            method (str): Thresholding method: one of ['static', 'pot', 'quantile', 'evt'].
            k (float): Multiplier for std in static thresholding (default: 3.0).
            quantile (float): Quantile level for quantile thresholding (default: 0.99).
            pot_q (float): Initial quantile for POT threshold estimation (default: 0.98).
            pot_level (float): False alarm rate for POT threshold (default: 0.01).
            evt_block_size (int): Block size for EVT (block maxima) fitting (default: 50).
            evt_fpr (float): Desired false positive rate in EVT (default: 0.01).
        """
        self.method = method
        self.k = k
        self.quantile = quantile
        self.pot_q = pot_q
        self.pot_level = pot_level
        self.evt_block_size = evt_block_size
        self.evt_fpr = evt_fpr
        self.threshold: Optional[float] = None

    def fit(self, train_scores: np.ndarray) -> None:
        """
        Fit the thresholding model on training anomaly scores.

        Args:
            train_scores (np.ndarray): Anomaly scores from normal training data.
        """
        if self.method == "static":
            mean = np.mean(train_scores)
            std = np.std(train_scores)
            self.threshold = mean + self.k * std

        elif self.method == "quantile":
            self.threshold = np.quantile(train_scores, self.quantile)

        elif self.method == "pot":
            initial_threshold = np.quantile(train_scores, self.pot_q)
            excesses = train_scores[train_scores > initial_threshold] - initial_threshold

            if len(excesses) < 5:
                raise RuntimeError("Too few exceedances for POT fitting.")

            c, loc, scale = genpareto.fit(excesses)
            threshold_excess = genpareto.ppf(1 - self.pot_level, c, loc=loc, scale=scale)
            self.threshold = initial_threshold + threshold_excess

        elif self.method == "evt":
            block_maxima = self._compute_block_maxima(train_scores)
            shape, loc, scale = genextreme.fit(block_maxima)
            self.threshold = genextreme.ppf(1 - self.evt_fpr, shape, loc=loc, scale=scale)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Classify scores as anomalies or not based on the computed threshold.

        Args:
            scores (np.ndarray): Anomaly scores to evaluate.

        Returns:
            np.ndarray: Boolean mask where True indicates anomaly.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")

        return scores > self.threshold

    def get_threshold(self) -> float:
        """
        Returns the computed threshold.

        Returns:
            float: The threshold value.
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before getting threshold.")
        return self.threshold

    def _compute_block_maxima(self, scores: np.ndarray) -> np.ndarray:
        """
        Helper to compute block maxima for EVT.

        Args:
            scores (np.ndarray): Anomaly scores.

        Returns:
            np.ndarray: Block maxima.
        """
        n_blocks = len(scores) // self.evt_block_size
        if n_blocks == 0:
            raise RuntimeError("Too few data points for EVT block maxima.")
        return np.max(
            scores[: n_blocks * self.evt_block_size].reshape(n_blocks, self.evt_block_size),
            axis=1
        )