/**
 * Veritas formula
 *
 *   Psi(x_i) = alpha * sum_j lambda_j * |(z_i - mu)^T v_j|^2
 *            + beta  * ||(I - P_i)(z_i - mu_i^k)||^2
 *            + gamma * sum_{j in N(i)} w_ij * ||z_i - z_j||^2
 *
 * Conventions (mirrors the Python package, except for eigenvector layout):
 *  - `eigenvectors` is a list of k vectors `v_j`, each of length d (orthonormal assumed).
 *  - `P` is a d x d matrix (array of rows). Default: V V^T (identity for the default basis).
 *  - `mu` defaults to the zero vector (or the fitted mean after `fit`).
 *  - the local mean defaults to the mean of the neighbours, or `mu` if there are none.
 *  - neighbour weights default to uniform 1 / |N(i)|.
 */
import {
  dot,
  identityRows,
  matVec,
  meanRows,
  sqNorm,
  sub,
  symmetricEigen,
  toMatrix,
  toVector,
  type Matrix,
  type Vector,
} from './linalg';

/** Immutable baseline hyper-parameters used by simple mode. */
export const DEFAULTS = Object.freeze({ alpha: 1, beta: 1, gamma: 1, nNeighbors: 5 });

export interface VeritasConfig {
  alpha?: number;
  beta?: number;
  gamma?: number;
  /** Nearest neighbours used automatically when reference `data` is available. */
  nNeighbors?: number;
  mu?: Vector;
  /** List of k eigenvectors v_j, each of length d. */
  eigenvectors?: Matrix;
  eigenvalues?: Vector;
  /** d x d projection matrix P_i. */
  P?: Matrix;
  /** Reference sample; rows are the z_j that neighbour indices refer to. */
  data?: Matrix;
}

export interface ScoreOptions {
  /** Row of the reference data that `x_i` came from (excluded from auto-kNN). */
  index?: number;
  mu?: Vector;
  /** Explicit local mean mu_i^k. */
  localMean?: Vector;
  P?: Matrix;
  eigenvectors?: Matrix;
  eigenvalues?: Vector;
  /** Neighbour indices N(i) into the reference data. */
  neighbors?: ArrayLike<number>;
  /** Neighbour vectors z_j given directly (alternative to indices). */
  neighborVectors?: Matrix;
  /** Weights w_ij, one per neighbour. */
  weights?: Vector;
  alpha?: number;
  beta?: number;
  gamma?: number;
  nNeighbors?: number;
}

export interface VeritasComponents {
  psi: number;
  /** alpha-weighted spectral term. */
  term1: number;
  /** beta-weighted residual term. */
  term2: number;
  /** gamma-weighted neighbourhood term. */
  term3: number;
}

function coef(v: unknown, name: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return v;
}

function positiveInt(v: unknown, name: string): number {
  if (typeof v !== 'number' || !Number.isInteger(v) || v < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return v;
}

export class VeritasEngine {
  alpha: number;
  beta: number;
  gamma: number;
  nNeighbors: number;
  mu: number[] | null;
  eigenvectors: number[][] | null;
  eigenvalues: number[] | null;
  P: number[][] | null;
  data: number[][] | null;

  constructor(config: VeritasConfig = {}) {
    this.alpha = coef(config.alpha ?? DEFAULTS.alpha, 'alpha');
    this.beta = coef(config.beta ?? DEFAULTS.beta, 'beta');
    this.gamma = coef(config.gamma ?? DEFAULTS.gamma, 'gamma');
    this.nNeighbors = positiveInt(config.nNeighbors ?? DEFAULTS.nNeighbors, 'nNeighbors');
    this.mu = config.mu !== undefined ? toVector(config.mu, 'mu') : null;
    this.eigenvectors =
      config.eigenvectors !== undefined ? toMatrix(config.eigenvectors, 'eigenvectors') : null;
    this.eigenvalues =
      config.eigenvalues !== undefined ? toVector(config.eigenvalues, 'eigenvalues') : null;
    this.P = config.P !== undefined ? toMatrix(config.P, 'P') : null;
    this.data = config.data !== undefined ? toMatrix(config.data, 'data') : null;
  }

  /** Estimate `mu` and the top-k eigenpairs from a sample (rows are observations). */
  fit(data: Matrix, nComponents?: number): this {
    const Z = toMatrix(data, 'data');
    const n = Z.length;
    const d = Z[0].length;
    const k = nComponents === undefined ? d : positiveInt(nComponents, 'nComponents');
    if (k > d) throw new RangeError('nComponents cannot exceed the data dimension');

    const mu = meanRows(Z);
    const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    for (const row of Z) {
      const c = sub(row, mu);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += c[i] * c[j];
    }
    const denom = Math.max(n - 1, 1);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] /= denom;

    const { values, vectors } = symmetricEigen(cov);
    this.mu = mu;
    this.eigenvalues = values.slice(0, k).map((v) => Math.max(v, 0));
    this.eigenvectors = vectors.slice(0, k);
    this.data = Z;
    return this;
  }

  /** Full breakdown of the score. Terms already include their weights. */
  components(x_i: Vector | number, opts: ScoreOptions = {}): VeritasComponents {
    const z = toVector(x_i, 'x_i');
    const d = z.length;

    const alpha = coef(opts.alpha ?? this.alpha, 'alpha');
    const beta = coef(opts.beta ?? this.beta, 'beta');
    const gamma = coef(opts.gamma ?? this.gamma, 'gamma');

    // global mean
    const mu =
      opts.mu !== undefined ? toVector(opts.mu, 'mu') : (this.mu ?? new Array<number>(d).fill(0));
    if (mu.length !== d) throw new RangeError(`mu has length ${mu.length}, expected ${d}`);

    // eigen-basis: list of k vectors v_j
    let V: number[][];
    let lamSrc: Vector | null | undefined;
    if (opts.eigenvectors !== undefined) {
      V = toMatrix(opts.eigenvectors, 'eigenvectors');
      lamSrc = opts.eigenvalues;
    } else {
      V = this.eigenvectors ?? identityRows(d);
      lamSrc = opts.eigenvalues ?? this.eigenvalues;
    }
    for (const v of V) {
      if (v.length !== d) throw new RangeError(`each eigenvector must have length ${d}`);
    }
    const lam = lamSrc ? toVector(lamSrc, 'eigenvalues') : new Array<number>(V.length).fill(1);
    if (lam.length !== V.length) {
      throw new RangeError(`expected ${V.length} eigenvalues, got ${lam.length}`);
    }

    // neighbourhood N(i)
    let Nz: number[][];
    if (opts.neighborVectors !== undefined) {
      Nz = opts.neighborVectors.length === 0 ? [] : toMatrix(opts.neighborVectors, 'neighborVectors');
      for (const r of Nz) {
        if (r.length !== d) throw new RangeError(`neighborVectors rows must have length ${d}`);
      }
    } else if (opts.neighbors !== undefined) {
      const idx = Array.from(opts.neighbors);
      if (idx.length > 0 && !this.data) {
        throw new Error('neighbor indices require fitted/reference data');
      }
      Nz = idx.map((i) => {
        if (!Number.isInteger(i) || i < 0 || i >= this.data!.length) {
          throw new RangeError(`neighbor index ${i} out of range`);
        }
        return this.data![i];
      });
    } else if (this.data) {
      if (this.data[0].length !== d) {
        throw new RangeError(`reference data has ${this.data[0].length} columns, expected ${d}`);
      }
      const nn = opts.nNeighbors === undefined ? this.nNeighbors : positiveInt(opts.nNeighbors, 'nNeighbors');
      const order = this.data
        .map((row, i) => ({ i, dist: sqNorm(sub(row, z)) }))
        .filter((e) => e.i !== opts.index)
        .sort((p, q) => p.dist - q.dist || p.i - q.i)
        .slice(0, nn);
      Nz = order.map((e) => this.data![e.i]);
    } else {
      Nz = [];
    }
    const m = Nz.length;

    // weights w_ij
    let w: number[];
    if (opts.weights === undefined) {
      w = m ? new Array<number>(m).fill(1 / m) : [];
    } else {
      w = Array.from(opts.weights);
      if (w.length !== m) throw new RangeError(`expected ${m} weights, got ${w.length}`);
      for (const x of w) {
        if (typeof x !== 'number' || !Number.isFinite(x)) {
          throw new TypeError('weights must contain only finite numbers');
        }
      }
    }

    // local mean mu_i^k
    let muLoc: number[];
    if (opts.localMean !== undefined) {
      muLoc = toVector(opts.localMean, 'localMean');
      if (muLoc.length !== d) throw new RangeError(`localMean has length ${muLoc.length}, expected ${d}`);
    } else {
      muLoc = m ? meanRows(Nz) : mu;
    }

    // term 1: spectral energy along eigen-directions
    const c = sub(z, mu);
    let spectral = 0;
    for (let j = 0; j < V.length; j++) {
      const p = dot(c, V[j]);
      spectral += lam[j] * p * p;
    }
    const term1 = alpha * spectral;

    // term 2: residual outside the projection subspace
    const r = sub(z, muLoc);
    const Pm = opts.P !== undefined ? toMatrix(opts.P, 'P') : this.P;
    let proj: number[];
    if (Pm) {
      if (Pm.length !== d || Pm[0].length !== d) throw new RangeError(`P must be ${d}x${d}`);
      proj = matVec(Pm, r);
    } else {
      proj = new Array<number>(d).fill(0);
      for (const v of V) {
        const coeff = dot(v, r);
        for (let i = 0; i < d; i++) proj[i] += coeff * v[i];
      }
    }
    const term2 = beta * sqNorm(sub(r, proj));

    // term 3: neighbourhood smoothness
    let smooth = 0;
    for (let j = 0; j < m; j++) smooth += w[j] * sqNorm(sub(z, Nz[j]));
    const term3 = gamma * smooth;

    const psi = term1 + term2 + term3;
    if (!Number.isFinite(psi)) throw new RangeError('Psi is not finite (overflow in inputs?)');
    return { psi, term1, term2, term3 };
  }

  /** Psi(x_i) as a plain number. */
  score(x_i: Vector | number, opts: ScoreOptions = {}): number {
    return this.components(x_i, opts).psi;
  }
}
