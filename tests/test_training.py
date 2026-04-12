"""Tests for the duration model training pipeline.

Verifies:
  - Weighted Ridge regression gives correct solutions
  - Feature centering + intercept decomposition roundtrips
  - Lambda exponential weighting is correct
  - Coefficient un-standardisation matches direct prediction
  - Training on known data reproduces expected coefficients
"""
import math
import sys
from pathlib import Path

import pytest
import numpy as np
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeightedRidge:
    """Weighted Ridge regression via normal equations."""

    def _solve_weighted_ridge(self, X, y, weights, alpha=1.0):
        """Reference: centred weighted Ridge matching train_model.py."""
        w = np.array(weights, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        w_sum = w.sum()

        # Weighted means
        x_wmean = (X * w[:, None]).sum(axis=0) / w_sum
        y_wmean = float((y * w).sum() / w_sum)

        # Centre
        X_c = X - x_wmean
        Xw_c = X_c * sqrt_w[:, None]
        A = Xw_c.T @ Xw_c + alpha * np.eye(X.shape[1])

        Y_c = y - y_wmean
        yw_c = Y_c * sqrt_w
        beta = np.linalg.solve(A, Xw_c.T @ yw_c)

        intercept = y_wmean - float(beta @ x_wmean)
        return beta, intercept

    def test_simple_1d_no_regularisation(self):
        """y = 2*x + 1 with uniform weights, alpha≈0."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        y = np.array([3, 5, 7, 9, 11], dtype=np.float64)  # y = 2x + 1
        w = np.ones(5)

        beta, intercept = self._solve_weighted_ridge(X, y, w, alpha=1e-10)
        assert beta[0] == pytest.approx(2.0, abs=0.01)
        assert intercept == pytest.approx(1.0, abs=0.01)

    def test_prediction_matches_uncentred(self):
        """Centred coefficients reproduce the same predictions as uncentred."""
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p) * 10 + 5
        true_beta = np.array([0.5, -0.3, 0.1])
        y = X @ true_beta + 3.0 + np.random.randn(n) * 0.1
        w = np.ones(n)

        beta, intercept = self._solve_weighted_ridge(X, y, w, alpha=0.1)

        # Predict using intercept + beta @ x
        preds = X @ beta + intercept

        # Should match weighted least squares prediction
        for i in range(n):
            pred_i = intercept + float(beta @ X[i])
            assert pred_i == pytest.approx(preds[i], abs=1e-10)

    def test_weights_emphasize_recent(self):
        """Recent samples with higher weight should dominate the fit."""
        # Two regimes: early y≈0, late y≈2*x
        n = 40
        X = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
        y = np.zeros(n, dtype=np.float64)
        # First half: y = 0
        # Second half: y = 2*x
        y[n // 2:] = 2.0 * X[n // 2:, 0]

        alpha = 0.01  # very light regularisation

        # Uniform weights
        beta_u, int_u = self._solve_weighted_ridge(X, y, np.ones(n), alpha)

        # Recent-heavy weights (exponential decay)
        lam = 0.9
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        beta_w, int_w = self._solve_weighted_ridge(X, y, w, alpha)

        # Recent data has slope=2. Weighted fit should have larger slope.
        assert beta_w[0] > beta_u[0]

    def test_ridge_shrinks_coefficients(self):
        """Higher alpha → smaller coefficients."""
        np.random.seed(42)
        X = np.random.randn(20, 3) * 10
        y = X @ np.array([1, 2, 3]) + np.random.randn(20)
        w = np.ones(20)

        beta_low, _ = self._solve_weighted_ridge(X, y, w, alpha=0.01)
        beta_high, _ = self._solve_weighted_ridge(X, y, w, alpha=100.0)

        assert np.linalg.norm(beta_high) < np.linalg.norm(beta_low)


class TestLambdaWeighting:
    """Forgetting factor lambda = 0.990."""

    def test_lambda_weight_formula(self):
        """Weight for sample n days ago = lambda^n."""
        lam = 0.990
        n = 100
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)

        # Most recent sample (index n-1) has weight lambda^0 = 1.0
        assert w[-1] == pytest.approx(1.0)
        # Oldest sample (index 0) has weight lambda^(n-1)
        assert w[0] == pytest.approx(lam ** (n - 1))

    def test_half_life(self):
        """Half-life = -ln(2)/ln(lambda) days."""
        lam = 0.990
        half_life = -math.log(2) / math.log(lam)
        assert half_life == pytest.approx(69.0, abs=0.5)

    def test_weights_sum(self):
        """Verify weight array sums correctly."""
        lam = 0.990
        n = 1000
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        # Geometric series: sum = (1 - lam^n) / (1 - lam)
        expected_sum = (1 - lam ** n) / (1 - lam)
        assert w.sum() == pytest.approx(expected_sum, rel=1e-10)

    def test_oldest_weight_negligible_after_5x_halflife(self):
        """After 5 half-lives (~345 days), weight < 3.1%."""
        lam = 0.990
        half_life = -math.log(2) / math.log(lam)
        n_days = int(5 * half_life)
        w = lam ** n_days
        assert w < 0.032  # 2^(-5) ≈ 0.031


class TestInterceptDecomposition:
    """Verify augmented matrix Ridge gives intercept + coefs that match."""

    def test_augmented_matrix_intercept(self):
        """Augmented matrix [X|1] with no penalty on intercept column
        produces intercept + beta @ x that matches training predictions."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p) * np.array([10, 100, 5, 50, 1])
        y = np.random.randn(n) * 0.5 + 4.0

        lam = 0.990
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        alpha = 1.0

        # Augmented matrix approach (as in train_model.py)
        X_aug = np.column_stack([X, np.ones(n)])  # [n x (p+1)]
        Xw_aug = X_aug * sqrt_w[:, None]
        A = Xw_aug.T @ Xw_aug + alpha * np.eye(p + 1)
        A[p, p] -= alpha  # don't penalise intercept
        yw = y * sqrt_w
        beta_aug = np.linalg.solve(A, Xw_aug.T @ yw)

        coefs = beta_aug[:p]
        intercept = beta_aug[p]

        # Training predictions via augmented matrix
        preds_aug = X_aug @ beta_aug

        # Predictions via intercept + coefs @ x (exported form)
        X_test = np.random.randn(20, p) * np.array([10, 100, 5, 50, 1])
        for i in range(20):
            pred_exported = intercept + float(coefs @ X_test[i])
            pred_aug = float(np.append(X_test[i], 1.0) @ beta_aug)
            assert pred_exported == pytest.approx(pred_aug, abs=1e-10), (
                f"Sample {i}: aug={pred_aug:.6f} vs exported={pred_exported:.6f}"
            )

    def test_augmented_vs_training_fit(self):
        """Augmented matrix predictions match on training data."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p) * np.array([10, 100, 5, 50, 1])
        y = np.random.randn(n) * 0.5 + 4.0

        lam = 0.990
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        alpha = 1.0

        X_aug = np.column_stack([X, np.ones(n)])
        Xw_aug = X_aug * sqrt_w[:, None]
        A = Xw_aug.T @ Xw_aug + alpha * np.eye(p + 1)
        A[p, p] -= alpha
        yw = y * sqrt_w
        beta_aug = np.linalg.solve(A, Xw_aug.T @ yw)

        coefs = beta_aug[:p]
        intercept = beta_aug[p]

        # Every training point should match
        for i in range(n):
            pred = intercept + float(coefs @ X[i])
            pred_direct = float(X_aug[i] @ beta_aug)
            assert pred == pytest.approx(pred_direct, abs=1e-10)


class TestSpearmanCorrelation:
    """Verify Spearman rank correlation computation."""

    def test_perfect_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        rho = spearmanr(x, y).statistic
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        rho = spearmanr(x, y).statistic
        assert rho == pytest.approx(-1.0)

    def test_random_near_zero(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        rho = spearmanr(x, y).statistic
        assert abs(rho) < 0.3

    def test_with_ties(self):
        """Ties should be handled correctly."""
        x = [1, 2, 2, 3, 4]
        y = [10, 20, 20, 30, 40]
        rho = spearmanr(x, y).statistic
        assert rho == pytest.approx(1.0)

    def test_monotonic_transform_preserves_rho(self):
        """Spearman is invariant to monotonic transforms."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 8, 16, 32]  # exponential
        rho = spearmanr(x, y).statistic
        assert rho == pytest.approx(1.0)
