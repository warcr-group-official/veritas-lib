# veritas-lib

Python and JavaScript/TypeScript implementations of the **Veritas formula**:

```
Ψ(xᵢ) = α · Σⱼ λⱼ·|(zᵢ − μ)ᵀvⱼ|²  +  β · ‖(I − Pᵢ)(zᵢ − μᵢᵏ)‖²  +  γ · Σ_{j∈N(i)} wᵢⱼ·‖zᵢ − zⱼ‖²
```

| Term | Meaning | Weight |
|------|---------|--------|
| 1 | spectral energy of `zᵢ − μ` along eigenvectors `vⱼ` (eigenvalues `λⱼ`) | `alpha` |
| 2 | residual of `zᵢ − μᵢᵏ` outside the projection `Pᵢ` | `beta` |
| 3 | weighted squared distance to neighbours `N(i)` | `gamma` |

**Simple-mode defaults:** `zᵢ = xᵢ`, `μ = 0`, standard basis with `λⱼ = 1`, `Pᵢ = VVᵀ` (identity → term 2 is 0),
local mean = mean of the neighbours (or `μ` if none), `wᵢⱼ = 1/|N(i)|`, `α = β = γ = 1`.

```
python/   Python package (numpy)        js/   TypeScript package (ESM, CJS, CDN bundle)
```

## Python

```bash
cd python
pip install -e .
```

```python
import numpy as np
import veritas
from veritas import VeritasEngine

# Simple mode
veritas.eval([1.0, 2.0, 3.0])      # 14.0   (veritas([1, 2, 3]) also works)

# Advanced mode
engine = VeritasEngine(alpha=2, beta=3, gamma=4)
engine.score(
    [3.0, 4.0],
    mu=[1, 1],
    eigenvectors=np.eye(2),            # (d, k): columns are v_j
    eigenvalues=[2.0, 0.5],
    P=[[1, 0], [0, 0]],                # (d, d)
    local_mean=[1, 2],
    neighbor_vectors=[[0, 0], [2, 2]],
    weights=[0.5, 1.0],
)                                      # 107.0

# Learn mu / eigenpairs from data; neighbours are found automatically (pass index= for a row of the data)
engine = VeritasEngine(n_neighbors=5).fit(Z, n_components=3)
engine.score(Z[0], index=0)
engine.components(Z[0], index=0)       # {'psi', 'term1', 'term2', 'term3'}
```

Tests: `python -m unittest discover -s tests -v` (no extra packages needed; `pytest` works too).

## JavaScript / TypeScript

```bash
cd js
npm install
npm run build      # dist/index.mjs, dist/index.cjs, dist/index.d.ts, dist/veritas.min.js
npm test           # builds, then runs tests with Node's built-in runner
```

```ts
import veritas, { VeritasEngine } from 'veritas-lib';
// CommonJS: const { veritas, VeritasEngine } = require('veritas-lib');

// Simple mode
veritas([1, 2, 3]);                    // 14

// Advanced mode
const engine = new VeritasEngine({ alpha: 2, beta: 3, gamma: 4 });
engine.score([3, 4], {
  mu: [1, 1],
  eigenvectors: [[1, 0], [0, 1]],      // list of k vectors v_j, each of length d
  eigenvalues: [2, 0.5],
  P: [[1, 0], [0, 0]],                 // d × d, array of rows
  localMean: [1, 2],
  neighborVectors: [[0, 0], [2, 2]],   // or neighbors: [indices] into engine data
  weights: [0.5, 1],
});                                    // 107
```

Browser / CDN (`veritas` and `VeritasEngine` are attached to `window`):

```html
<script src="https://cdn.jsdelivr.net/npm/veritas-lib/dist/veritas.min.js"></script>
<script>
  console.log(veritas([1, 2, 3]));                       // 14
  console.log(new VeritasEngine({ alpha: 2 }).score([1, 2, 3]));
</script>
```

Simple mode builds a fresh engine from frozen defaults on every call, so nothing global is shared or mutable.

## License

MIT
