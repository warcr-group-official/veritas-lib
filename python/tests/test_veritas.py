"""Tests for veritas-lib. Pure ``unittest`` so they run with no extra installs:

    python -m unittest discover -s tests -v

(``pytest`` also collects them.)
"""
import math
import unittest

import numpy as np

import veritas
from veritas import VeritasEngine


def advanced_inputs():
    return dict(
        mu=[1.0, 1.0],
        eigenvectors=np.eye(2),
        eigenvalues=[2.0, 0.5],
        P=[[1.0, 0.0], [0.0, 0.0]],
        local_mean=[1.0, 2.0],
        neighbor_vectors=[[0.0, 0.0], [2.0, 2.0]],
        weights=[0.5, 1.0],
    )


class SimpleModeTests(unittest.TestCase):
    def test_returns_valid_float(self):
        psi = veritas.eval([1.0, 2.0, 3.0])
        self.assertIsInstance(psi, float)
        self.assertTrue(math.isfinite(psi))
        # mu=0, identity basis (lambda=1), P=I, no neighbours -> ||z||^2
        self.assertAlmostEqual(psi, 14.0)

    def test_module_is_callable_and_accepts_numpy(self):
        self.assertAlmostEqual(veritas([1, 2, 3]), 14.0)
        self.assertAlmostEqual(veritas.eval(np.array([0.5, -1.5])), 2.5)

    def test_isolated_between_calls(self):
        before = dict(veritas.DEFAULTS)
        self.assertAlmostEqual(veritas.eval([1, 2, 3], alpha=10.0), 140.0)
        self.assertAlmostEqual(veritas.eval([1, 2, 3]), 14.0)  # override did not leak
        self.assertEqual(dict(veritas.DEFAULTS), before)
        with self.assertRaises(TypeError):
            veritas.DEFAULTS["alpha"] = 99  # immutable


class AdvancedModeTests(unittest.TestCase):
    def test_custom_weights_and_matrices(self):
        z = [3.0, 4.0]
        engine = VeritasEngine(alpha=2.0, beta=3.0, gamma=4.0)
        parts = engine.components(z, **advanced_inputs())
        # term1: (2*4 + 0.5*9) * 2 = 25 ; term2: ||(0,2)||^2 * 3 = 12 ; term3: 17.5 * 4 = 70
        self.assertAlmostEqual(parts["term1"], 25.0)
        self.assertAlmostEqual(parts["term2"], 12.0)
        self.assertAlmostEqual(parts["term3"], 70.0)
        self.assertAlmostEqual(parts["psi"], 107.0)
        self.assertAlmostEqual(engine.score(z, **advanced_inputs()), 107.0)
        self.assertAlmostEqual(engine(z, **advanced_inputs()), 107.0)

    def test_each_coefficient_scales_only_its_term(self):
        cases = [
            (0, 0, 0, 0.0),
            (1, 0, 0, 12.5),
            (0, 1, 0, 4.0),
            (0, 0, 1, 17.5),
            (1, 1, 1, 34.0),
        ]
        for alpha, beta, gamma, expected in cases:
            with self.subTest(alpha=alpha, beta=beta, gamma=gamma):
                engine = VeritasEngine(alpha=alpha, beta=beta, gamma=gamma)
                self.assertAlmostEqual(engine.score([3.0, 4.0], **advanced_inputs()), expected)

    def test_custom_projection_changes_term2(self):
        base = dict(mu=[0, 0], local_mean=[0, 0])
        z = [3.0, 4.0]
        identity = VeritasEngine(beta=1).components(z, P=np.eye(2), **base)
        zero = VeritasEngine(beta=1).components(z, P=np.zeros((2, 2)), **base)
        self.assertAlmostEqual(identity["term2"], 0.0)
        self.assertAlmostEqual(zero["term2"], 25.0)

    def test_per_call_overrides_beat_constructor_values(self):
        engine = VeritasEngine(alpha=1.0, mu=[1, 1])
        self.assertEqual(engine.score([3, 4], alpha=0.0, beta=0.0, gamma=0.0), 0.0)


class FitAndNeighbourTests(unittest.TestCase):
    def test_fit_estimates_mean_and_eigenpairs(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(50, 4)) @ np.diag([3.0, 2.0, 1.0, 0.5])
        engine = VeritasEngine().fit(Z, n_components=2)
        np.testing.assert_allclose(engine.mu, Z.mean(axis=0))
        self.assertEqual(engine.eigenvectors.shape, (4, 2))
        self.assertGreaterEqual(engine.eigenvalues[0], engine.eigenvalues[1])
        self.assertGreaterEqual(engine.eigenvalues[1], 0.0)
        psi = engine.score(Z[0], index=0)
        self.assertIsInstance(psi, float)
        self.assertTrue(math.isfinite(psi))

    def test_explicit_neighbor_indices_use_reference_data(self):
        Z = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [5.0, 5.0]])
        engine = VeritasEngine(alpha=0, beta=0, gamma=1, data=Z)
        got = engine.score([1.0, 1.0], neighbors=[0, 1], weights=[1.0, 2.0])
        self.assertAlmostEqual(got, 6.0)  # 1*2 + 2*2

    def test_auto_knn_excludes_self_when_index_given(self):
        Z = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        engine = VeritasEngine(alpha=0, beta=0, gamma=1, n_neighbors=1).fit(Z)
        self.assertAlmostEqual(engine.score(Z[0]), 0.0)  # nearest is itself
        self.assertAlmostEqual(engine.score(Z[0], index=0), 1.0)  # nearest is Z[1]


class ValidationTests(unittest.TestCase):
    def test_invalid_vectors_are_rejected(self):
        for bad in ([], [1.0, float("nan")], [[1, 2], [3, 4]], ["a", "b"]):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    veritas.eval(bad)

    def test_dimension_mismatches_are_rejected(self):
        engine = VeritasEngine()
        with self.assertRaises(ValueError):
            engine.score([1, 2, 3], mu=[0, 0])
        with self.assertRaises(ValueError):
            engine.score([1, 2, 3], P=np.eye(2))
        with self.assertRaises(ValueError):
            engine.score([1, 2, 3], neighbor_vectors=[[1, 2]], weights=[1.0])
        with self.assertRaises(ValueError):
            engine.score([1, 2], neighbor_vectors=[[1, 2]], weights=[1.0, 2.0])

    def test_bad_hyperparameters_are_rejected(self):
        with self.assertRaises(ValueError):
            VeritasEngine(alpha=float("inf"))
        with self.assertRaises(ValueError):
            VeritasEngine(n_neighbors=0)


if __name__ == "__main__":
    unittest.main()
