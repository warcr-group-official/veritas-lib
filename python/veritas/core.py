"""Reference implementation of the Veritas formula.

    Psi(x_i) = alpha * sum_{j=1..k} lambda_j * |(z_i - mu)^T v_j|^2
             + beta  * ||(I - P_i)(z_i - mu_i^k)||_2^2
             + gamma * sum_{j in N(i)} w_ij * ||z_i - z_j||_2^2

Conventions
-----------
* ``z_i`` is the input vector ``x_i`` itself.
* ``eigenvectors`` is a ``(d, k)`` array whose *columns* are ``v_j`` (the layout
  returned by ``numpy.linalg.eigh``). Columns are assumed orthonormal.
* ``P`` is a ``(d, d)`` projection matrix. When omitted it defaults to
  ``V V^T`` (identity for the default full standard basis, so term 2 is 0).
* ``mu`` defaults to the zero vector (or the fitted mean after :meth:`fit`).
* ``mu_i^k`` (local mean) defaults to the mean of the neighbours, or ``mu`` if
  there are none.
* Neighbour weights ``w_ij`` default to uniform ``1 / |N(i)|``.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Optional

import numpy as np

#: Immutable baseline hyper-parameters used by simple mode.
DEFAULTS = MappingProxyType(
    {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "n_neighbors": 5}
)


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_float_array(x: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc


def _as_vector(x: Any, name: str) -> np.ndarray:
    a = _as_float_array(x, name)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D numeric vector")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite numbers")
    return a


def _as_matrix(x: Any, name: str, allow_vector: bool = False) -> np.ndarray:
    a = _as_float_array(x, name)
    if allow_vector and a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D numeric matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite numbers")
    return a


def _coef(v: Any, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not np.isfinite(f):
        raise ValueError(f"{name} must be a finite number")
    return f


def _positive_int(v: Any, name: str) -> int:
    try:
        i = int(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if i < 1 or i != v:
        raise ValueError(f"{name} must be a positive integer")
    return i


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class VeritasEngine:
    """Configurable evaluator of the Veritas score ``Psi``.

    Parameters
    ----------
    alpha, beta, gamma:
        Weights of the three terms.
    n_neighbors:
        Number of nearest neighbours used when ``data`` is available and no
        explicit neighbours are supplied.
    mu, eigenvectors, eigenvalues, P, data:
        Optional defaults for the global mean, the ``(d, k)`` eigenvector
        matrix, the ``k`` eigenvalues, the ``(d, d)`` projection matrix and the
        reference sample (rows are ``z_j``). Any of them can be overridden per
        call.
    """

    def __init__(
        self,
        alpha: float = DEFAULTS["alpha"],
        beta: float = DEFAULTS["beta"],
        gamma: float = DEFAULTS["gamma"],
        *,
        n_neighbors: int = DEFAULTS["n_neighbors"],
        mu: Optional[Any] = None,
        eigenvectors: Optional[Any] = None,
        eigenvalues: Optional[Any] = None,
        P: Optional[Any] = None,
        data: Optional[Any] = None,
    ) -> None:
        self.alpha = _coef(alpha, "alpha")
        self.beta = _coef(beta, "beta")
        self.gamma = _coef(gamma, "gamma")
        self.n_neighbors = _positive_int(n_neighbors, "n_neighbors")
        self.mu = None if mu is None else _as_vector(mu, "mu")
        self.eigenvectors = (
            None
            if eigenvectors is None
            else _as_matrix(eigenvectors, "eigenvectors", allow_vector=True)
        )
        self.eigenvalues = (
            None if eigenvalues is None else _as_vector(eigenvalues, "eigenvalues")
        )
        self.P = None if P is None else _as_matrix(P, "P")
        self.data = None if data is None else _as_matrix(data, "data")

    # ------------------------------------------------------------------ fit
    def fit(self, data: Any, n_components: Optional[int] = None) -> "VeritasEngine":
        """Estimate ``mu`` and the top-``k`` eigenpairs from a sample.

        ``data`` has shape ``(n, d)``. The sample is also stored so that
        neighbours can be found automatically (pass ``index=`` when scoring a
        row of the sample so it is not its own neighbour).
        """
        Z = _as_matrix(data, "data")
        n, d = Z.shape
        k = d if n_components is None else _positive_int(n_components, "n_components")
        if k > d:
            raise ValueError("n_components cannot exceed the data dimension")
        mu = Z.mean(axis=0)
        C = Z - mu
        cov = (C.T @ C) / max(n - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:k]
        self.mu = mu
        self.eigenvalues = np.clip(vals[order], 0.0, None)
        self.eigenvectors = vecs[:, order]
        self.data = Z
        return self

    # ----------------------------------------------------------- components
    def components(
        self,
        x_i: Any,
        *,
        index: Optional[int] = None,
        mu: Optional[Any] = None,
        local_mean: Optional[Any] = None,
        P: Optional[Any] = None,
        eigenvectors: Optional[Any] = None,
        eigenvalues: Optional[Any] = None,
        neighbors: Optional[Any] = None,
        neighbor_vectors: Optional[Any] = None,
        weights: Optional[Any] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        n_neighbors: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return ``{"psi", "term1", "term2", "term3"}`` (terms include their weights)."""
        z = _as_vector(x_i, "x_i")
        d = z.size

        a = self.alpha if alpha is None else _coef(alpha, "alpha")
        b = self.beta if beta is None else _coef(beta, "beta")
        g = self.gamma if gamma is None else _coef(gamma, "gamma")

        # global mean
        if mu is not None:
            mu_v = _as_vector(mu, "mu")
        elif self.mu is not None:
            mu_v = self.mu
        else:
            mu_v = np.zeros(d)
        if mu_v.size != d:
            raise ValueError(f"mu has length {mu_v.size}, expected {d}")

        # eigen-basis (columns of V are v_j)
        if eigenvectors is not None:
            V = _as_matrix(eigenvectors, "eigenvectors", allow_vector=True)
            lam_src = eigenvalues
        else:
            V = self.eigenvectors if self.eigenvectors is not None else np.eye(d)
            lam_src = eigenvalues if eigenvalues is not None else self.eigenvalues
        if V.shape[0] != d:
            raise ValueError(f"eigenvectors must have {d} rows, got {V.shape[0]}")
        k = V.shape[1]
        lam = np.ones(k) if lam_src is None else _as_vector(lam_src, "eigenvalues")
        if lam.size != k:
            raise ValueError(f"expected {k} eigenvalues, got {lam.size}")

        # neighbourhood N(i)
        if neighbor_vectors is not None:
            Nz = _as_matrix(neighbor_vectors, "neighbor_vectors")
            if Nz.shape[1] != d:
                raise ValueError(f"neighbor_vectors must have {d} columns")
        elif neighbors is not None:
            idx = np.asarray(neighbors)
            if idx.size == 0:
                Nz = np.empty((0, d))
            else:
                if self.data is None:
                    raise ValueError("neighbor indices require fitted/reference data")
                if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
                    raise TypeError("neighbors must be a 1-D sequence of integer indices")
                if idx.min() < 0 or idx.max() >= self.data.shape[0]:
                    raise IndexError("neighbor index out of range")
                Nz = self.data[idx]
        elif self.data is not None:
            if self.data.shape[1] != d:
                raise ValueError(f"reference data has {self.data.shape[1]} columns, expected {d}")
            nn = self.n_neighbors if n_neighbors is None else _positive_int(n_neighbors, "n_neighbors")
            dist = np.sum((self.data - z) ** 2, axis=1)
            order = np.argsort(dist, kind="stable")
            if index is not None:
                order = order[order != int(index)]
            Nz = self.data[order[: min(nn, order.size)]]
        else:
            Nz = np.empty((0, d))
        m = Nz.shape[0]

        # weights w_ij
        if weights is None:
            w = np.full(m, 1.0 / m) if m else np.empty(0)
        else:
            w = _as_float_array(weights, "weights").reshape(-1)
            if w.size != m:
                raise ValueError(f"expected {m} weights, got {w.size}")
            if not np.all(np.isfinite(w)):
                raise ValueError("weights must contain only finite numbers")

        # local mean mu_i^k
        if local_mean is not None:
            mu_loc = _as_vector(local_mean, "local_mean")
            if mu_loc.size != d:
                raise ValueError(f"local_mean has length {mu_loc.size}, expected {d}")
        elif m:
            mu_loc = Nz.mean(axis=0)
        else:
            mu_loc = mu_v

        # term 1: spectral energy along the eigen-directions
        c = z - mu_v
        term1 = a * float(np.dot(lam, (V.T @ c) ** 2))

        # term 2: residual outside the projection subspace
        r = z - mu_loc
        P_src = P if P is not None else self.P
        if P_src is None:
            proj = V @ (V.T @ r)
        else:
            Pm = _as_matrix(P_src, "P")
            if Pm.shape != (d, d):
                raise ValueError(f"P must be {d}x{d}, got {Pm.shape[0]}x{Pm.shape[1]}")
            proj = Pm @ r
        resid = r - proj
        term2 = b * float(np.dot(resid, resid))

        # term 3: neighbourhood smoothness
        term3 = g * float(np.dot(w, np.sum((Nz - z) ** 2, axis=1))) if m else 0.0

        psi = term1 + term2 + term3
        if not np.isfinite(psi):
            raise FloatingPointError("Psi is not finite (overflow in inputs?)")
        return {"psi": psi, "term1": term1, "term2": term2, "term3": term3}

    # ---------------------------------------------------------------- score
    def score(self, x_i: Any, **kwargs: Any) -> float:
        """Return ``Psi(x_i)`` as a float. Accepts the same keywords as :meth:`components`."""
        return float(self.components(x_i, **kwargs)["psi"])

    __call__ = score

    def __repr__(self) -> str:
        return (
            f"VeritasEngine(alpha={self.alpha}, beta={self.beta}, "
            f"gamma={self.gamma}, n_neighbors={self.n_neighbors})"
        )
