# ⚖️ veritas-lib

[![npm version](https://img.shields.io/npm/v/veritas-lib.svg?color=cb3837&style=flat-square)](https://www.npmjs.com/package/veritas-lib)
[![PyPI version](https://img.shields.io/pypi/v/veritas-lib.svg?color=3775a9&style=flat-square)](https://pypi.org/project/veritas-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg?style=flat-square)](https://www.typescriptlang.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/)

High-performance, dual-language implementation of the **Veritas Formula** for high-dimensional anomaly detection, manifold alignment, spectral energy evaluation, and spatial regularization in **Python** (NumPy) and **JavaScript / TypeScript** (Node.js, ESM, CJS, Browser).

---

## 📐 Mathematical Formulation

The **Veritas score** $\Psi(x_i)$ balances global spectral variance, local manifold subspace projection residuals, and neighborhood graph smoothness:

$$\Psi(x_i) = \alpha \sum_{j} \lambda_j \left\vert (z_i - \mu)^T v_j \right\vert^2 + \beta \left\Vert (I - P_i)(z_i - \mu_i^k) \right\Vert^2 + \gamma \sum_{j \in \mathcal{N}(i)} w_{ij} \left\Vert z_i - z_j \right\Vert^2$$

### Formula Component Decomposition

| Term | Mathematical Definition | Physical / Geometric Interpretation | Default Weight |
| :--- | :--- | :--- | :--- |
| **Term 1** | $\alpha \sum_{j} \lambda_j \left\vert (z_i - \mu)^T v_j \right\vert^2$ | **Spectral Energy**: Weighted projection magnitude along principal covariance eigenvectors $v_j$ with eigenvalues $\lambda_j$. | $\alpha = 1.0$ |
| **Term 2** | $\beta \left\Vert (I - P_i)(z_i - \mu_i^k) \right\Vert^2$ | **Subspace Residual**: Orthogonal distance of mean-centered vector $z_i - \mu_i^k$ outside local subspace projection matrix $P_i$. | $\beta = 1.0$ |
| **Term 3** | $\gamma \sum_{j \in \mathcal{N}(i)} w_{ij} \left\Vert z_i - z_j \right\Vert^2$ | **Graph Smoothness**: Weighted Euclidean spatial variance relative to neighboring graph nodes $j \in \mathcal{N}(i)$. | $\gamma = 1.0$ |

#### Simple-Mode Defaults

When invoked without advanced parameter arrays:

* Input normalization: $z_i = x_i$, global mean $\mu = \mathbf{0}$.
* Standard basis eigenvectors $v_j = e_j$ with unit eigenvalues $\lambda_j = 1$.
* Subspace matrix $P_i = VV^T = I$ (identity matrix, rendering Term 2 residual = $0$).
* Local mean $\mu_i^k = \text{mean}(\mathcal{N}(i))$ (or $\mu$ if no neighbors).
* Uniform weights $w_{ij} = \frac{1}{\vert \mathcal{N}(i) \vert}$.
* Hyperparameters $\alpha = \beta = \gamma = 1$.

---

## 🚀 Quick Installation

### Python

```bash
pip install veritas-lib
```

### JavaScript / TypeScript

```bash
npm install veritas-lib
```

---

## 🐍 Python Usage

### 1. Simple Mode

Quick zero-configuration evaluation using default frozen identity transformations:

```python
import veritas

# Functional interface (eval or veritas call)
score = veritas.eval([1.0, 2.0, 3.0])
print(score)  # Output: 14.0
```

### 2. Advanced Custom Parameter Scoring

Manually pass custom eigenvectors, projection matrices, and neighbor weights:

```python
import numpy as np
from veritas import VeritasEngine

engine = VeritasEngine(alpha=2.0, beta=3.0, gamma=4.0)

score = engine.score(
    x=[3.0, 4.0],
    mu=[1.0, 1.0],
    eigenvectors=np.eye(2),             # (d, k): columns are eigenvectors v_j
    eigenvalues=[2.0, 0.5],             # Eigenvalues lambda_j
    P=[[1.0, 0.0], [0.0, 0.0]],         # (d, d) Subspace projection matrix
    local_mean=[1.0, 2.0],
    neighbor_vectors=[[0.0, 0.0], [2.0, 2.0]],
    weights=[0.5, 1.0],
)
print(score)  # Output: 107.0
```

### 3. Data-Driven Fitting & Component Analysis

Fit global statistics (μ, eigenpairs, k-NN graph) automatically from a dataset matrix:

```python
import numpy as np
from veritas import VeritasEngine

# Sample dataset (N samples, D dimensions)
Z = np.random.randn(100, 5)

# Fit engine on dataset Z with 5 nearest neighbors and 3 components
engine = VeritasEngine(n_neighbors=5).fit(Z, n_components=3)

# Evaluate specific sample score
score = engine.score(Z[0], index=0)

# Decompose score into individual formula terms
components = engine.components(Z[0], index=0)
print(components)
# Output: {'psi': float, 'term1': float, 'term2': float, 'term3': float}
```

---

## ⚡ JavaScript / TypeScript Usage

### 1. Node.js / Bundlers (ESM & CommonJS)

```typescript
import veritas, { VeritasEngine } from 'veritas-lib';
// CommonJS alternative: const { veritas, VeritasEngine } = require('veritas-lib');

// Simple mode evaluation
console.log(veritas([1, 2, 3])); // Output: 14

// Advanced mode evaluation
const engine = new VeritasEngine({ alpha: 2, beta: 3, gamma: 4 });

const score = engine.score([3, 4], {
  mu: [1, 1],
  eigenvectors: [[1, 0], [0, 1]],        // Array of k vectors v_j
  eigenvalues: [2, 0.5],
  P: [[1, 0], [0, 0]],                   // d × d array of rows
  localMean: [1, 2],
  neighborVectors: [[0, 0], [2, 2]],     // Or pass `neighbors: [indices]` into fitted data
  weights: [0.5, 1],
});

console.log(score); // Output: 107
```

### 2. Browser via CDN

Including `veritas.min.js` exposes global `veritas` and `VeritasEngine` objects on `window`:

```html
<script src="https://cdn.jsdelivr.net/npm/veritas-lib/dist/veritas.min.js"></script>
<script>
  // Simple functional score
  console.log(veritas([1, 2, 3])); // 14

  // Engine instance score
  const engine = new VeritasEngine({ alpha: 2.0 });
  console.log(engine.score([1, 2, 3]));
</script>
```

---

## 🛠️ Local Development & Testing

### Python Development

```bash
cd python
pip install -e .

# Run test suite
python -m unittest discover -s tests -v
# or with pytest:
pytest
```

### TypeScript Development

```bash
cd js
npm install

# Build ESM, CJS, and CDN minified bundles
npm run build

# Run automated test runner
npm test
```

---

## 📄 License

This project is licensed under the terms of the MIT License.
