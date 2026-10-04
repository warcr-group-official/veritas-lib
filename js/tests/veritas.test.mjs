// Run with `npm test` (builds first, then uses Node's built-in test runner).
import test from 'node:test';
import assert from 'node:assert/strict';
import veritasDefault, { veritas, VeritasEngine, DEFAULTS, symmetricEigen } from '../dist/index.mjs';

const close = (actual, expected, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${actual} ≈ ${expected}`);

const advancedInputs = () => ({
  mu: [1, 1],
  eigenvectors: [[1, 0], [0, 1]], // list of v_j
  eigenvalues: [2, 0.5],
  P: [[1, 0], [0, 0]],
  localMean: [1, 2],
  neighborVectors: [[0, 0], [2, 2]],
  weights: [0.5, 1],
});

// ------------------------------------------------------------------ simple mode
test('simple mode returns a valid finite number', () => {
  const psi = veritas([1, 2, 3]);
  assert.equal(typeof psi, 'number');
  assert.ok(Number.isFinite(psi));
  close(psi, 14); // mu=0, identity basis, P=I, no neighbours -> ||z||^2
});

test('default export is the same function; typed arrays and scalars work', () => {
  assert.equal(veritasDefault, veritas);
  close(veritas(new Float64Array([0.5, -1.5])), 2.5);
  close(veritas(3), 9);
});

test('simple mode is isolated: overrides never leak and defaults are frozen', () => {
  close(veritas([1, 2, 3], { alpha: 10 }), 140);
  close(veritas([1, 2, 3]), 14);
  assert.deepEqual({ ...DEFAULTS }, { alpha: 1, beta: 1, gamma: 1, nNeighbors: 5 });
  assert.throws(() => {
    DEFAULTS.alpha = 99; // ES modules are strict: writing to a frozen object throws
  }, TypeError);
  assert.equal(DEFAULTS.alpha, 1);
});

// ---------------------------------------------------------------- advanced mode
test('advanced mode applies alpha/beta/gamma and custom matrices', () => {
  const engine = new VeritasEngine({ alpha: 2, beta: 3, gamma: 4 });
  const parts = engine.components([3, 4], advancedInputs());
  // term1: (2*4 + 0.5*9)*2 = 25 ; term2: ||(0,2)||^2*3 = 12 ; term3: 17.5*4 = 70
  close(parts.term1, 25);
  close(parts.term2, 12);
  close(parts.term3, 70);
  close(parts.psi, 107);
  close(engine.score([3, 4], advancedInputs()), 107);
});

test('each coefficient scales only its own term', () => {
  const cases = [
    [0, 0, 0, 0],
    [1, 0, 0, 12.5],
    [0, 1, 0, 4],
    [0, 0, 1, 17.5],
    [1, 1, 1, 34],
  ];
  for (const [alpha, beta, gamma, expected] of cases) {
    close(new VeritasEngine({ alpha, beta, gamma }).score([3, 4], advancedInputs()), expected);
  }
});

test('custom projection matrix changes term2', () => {
  const base = { mu: [0, 0], localMean: [0, 0] };
  const identity = new VeritasEngine({ beta: 1 }).components([3, 4], { ...base, P: [[1, 0], [0, 1]] });
  const zero = new VeritasEngine({ beta: 1 }).components([3, 4], { ...base, P: [[0, 0], [0, 0]] });
  close(identity.term2, 0);
  close(zero.term2, 25);
});

test('constructor values act as defaults and per-call options override them', () => {
  const engine = new VeritasEngine({ alpha: 1, mu: [1, 1] });
  assert.equal(engine.score([3, 4], { alpha: 0, beta: 0, gamma: 0 }), 0);
});

// ------------------------------------------------------- fit / neighbour search
test('symmetricEigen reconstructs the matrix', () => {
  const A = [[4, 1, 2], [1, 3, 0], [2, 0, 5]];
  const { values, vectors } = symmetricEigen(A);
  assert.ok(values[0] >= values[1] && values[1] >= values[2]);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += values[k] * vectors[k][i] * vectors[k][j];
      close(s, A[i][j], 1e-9);
    }
  }
});

test('fit estimates the mean and top eigenpairs', () => {
  const Z = [[1, 0], [-1, 0], [0, 2], [0, -2]];
  const engine = new VeritasEngine().fit(Z);
  close(engine.mu[0], 0);
  close(engine.mu[1], 0);
  close(engine.eigenvalues[0], 8 / 3);
  close(engine.eigenvalues[1], 2 / 3);
  close(Math.abs(engine.eigenvectors[0][1]), 1); // dominant direction is y
  const top1 = new VeritasEngine().fit(Z, 1);
  assert.equal(top1.eigenvectors.length, 1);
  assert.ok(Number.isFinite(top1.score([1, 1], { index: 0 })));
});

test('explicit neighbour indices use the reference data', () => {
  const Z = [[0, 0], [2, 0], [0, 2], [5, 5]];
  const engine = new VeritasEngine({ alpha: 0, beta: 0, gamma: 1, data: Z });
  close(engine.score([1, 1], { neighbors: [0, 1], weights: [1, 2] }), 6); // 1*2 + 2*2
});

test('auto k-NN excludes the point itself when `index` is given', () => {
  const Z = [[0, 0], [1, 0], [10, 0]];
  const engine = new VeritasEngine({ alpha: 0, beta: 0, gamma: 1, nNeighbors: 1 }).fit(Z);
  close(engine.score(Z[0]), 0); // nearest is itself
  close(engine.score(Z[0], { index: 0 }), 1); // nearest is Z[1]
});

// ------------------------------------------------------------------- validation
test('invalid input is rejected', () => {
  assert.throws(() => veritas([]));
  assert.throws(() => veritas([1, NaN]));
  assert.throws(() => veritas([1, 'a']));
  assert.throws(() => veritas([1, 2, 3], { mu: [0, 0] }), RangeError);
  assert.throws(() => veritas([1, 2, 3], { P: [[1, 0], [0, 1]] }), RangeError);
  assert.throws(() => veritas([1, 2], { neighborVectors: [[1, 2]], weights: [1, 2] }), RangeError);
  assert.throws(() => veritas([1, 2], { neighbors: [0] }), Error); // no reference data
  assert.throws(() => new VeritasEngine({ alpha: Infinity }), TypeError);
  assert.throws(() => new VeritasEngine({ nNeighbors: 0 }), RangeError);
});
