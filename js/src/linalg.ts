/** Small dependency-free linear-algebra helpers used by the engine. */

export type Vector = ArrayLike<number>;
export type Matrix = ReadonlyArray<ArrayLike<number>>;

/** Validate and copy a numeric vector (a bare number becomes a length-1 vector). */
export function toVector(x: Vector | number, name = 'vector'): number[] {
  if (x === null || x === undefined) throw new TypeError(`${name} is required`);
  const arr = typeof x === 'number' ? [x] : Array.from(x);
  if (arr.length === 0) throw new RangeError(`${name} must be a non-empty numeric vector`);
  for (const v of arr) {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new TypeError(`${name} must contain only finite numbers`);
    }
  }
  return arr;
}

/** Validate and copy a matrix given as an array of equal-length rows. */
export function toMatrix(m: Matrix, name = 'matrix'): number[][] {
  if (m === null || m === undefined || typeof m.length !== 'number' || m.length === 0) {
    throw new TypeError(`${name} must be a non-empty array of rows`);
  }
  const rows: number[][] = [];
  for (let i = 0; i < m.length; i++) rows.push(toVector(m[i], `${name}[${i}]`));
  const width = rows[0].length;
  for (const r of rows) {
    if (r.length !== width) throw new RangeError(`${name} rows must all have the same length`);
  }
  return rows;
}

export const dot = (a: number[], b: number[]): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export const sub = (a: number[], b: number[]): number[] => a.map((v, i) => v - b[i]);

export const sqNorm = (a: number[]): number => dot(a, a);

export const matVec = (M: number[][], v: number[]): number[] => M.map((row) => dot(row, v));

export function meanRows(rows: number[][]): number[] {
  const out = new Array<number>(rows[0].length).fill(0);
  for (const r of rows) for (let i = 0; i < out.length; i++) out[i] += r[i];
  return out.map((v) => v / rows.length);
}

export function identityRows(n: number): number[][] {
  return Array.from({ length: n }, (_, i) => {
    const r = new Array<number>(n).fill(0);
    r[i] = 1;
    return r;
  });
}

/**
 * Eigen-decomposition of a real symmetric matrix (cyclic Jacobi).
 * Returns eigenvalues in descending order; `vectors[j]` is the unit
 * eigenvector for `values[j]`.
 */
export function symmetricEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v = identityRows(n);

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => v.map((row) => row[i])),
  };
}
