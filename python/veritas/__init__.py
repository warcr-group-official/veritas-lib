"""veritas-lib: the Veritas formula.

Simple mode::

    import veritas
    veritas.eval([1.0, 2.0, 3.0])      # float
    veritas([1.0, 2.0, 3.0])           # the module itself is callable

Advanced mode::

    from veritas import VeritasEngine
    engine = VeritasEngine(alpha=2, beta=3, gamma=4)
    engine.score(x, P=P, eigenvectors=V, eigenvalues=lam, neighbors=..., weights=w)
"""
import sys as _sys
import types as _types
from typing import Any

from .core import DEFAULTS, VeritasEngine

__version__ = "0.1.0"
__all__ = ["VeritasEngine", "DEFAULTS", "eval", "score", "__version__"]


def eval(x_i: Any, **kwargs: Any) -> float:  # noqa: A001 - public API name
    """Simple mode: ``Psi(x_i)`` with safe baseline defaults.

    A fresh engine is built from the immutable :data:`DEFAULTS` on every call,
    so no state is shared or mutated between calls.
    """
    return VeritasEngine().score(x_i, **kwargs)


score = eval


class _CallableModule(_types.ModuleType):
    def __call__(self, x_i: Any, **kwargs: Any) -> float:
        return eval(x_i, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
