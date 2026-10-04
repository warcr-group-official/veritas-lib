import { VeritasEngine, DEFAULTS, type ScoreOptions } from './engine';
import type { Vector } from './linalg';

/**
 * Simple mode: Psi(x_i) with safe baseline defaults.
 *
 * A brand-new engine is built from the frozen `DEFAULTS` on every call, so
 * nothing is shared or mutated between calls (and nothing is exposed for
 * tampering via DevTools).
 */
export function veritas(x_i: Vector | number, options: ScoreOptions = {}): number {
  return new VeritasEngine().score(x_i, options);
}

export { VeritasEngine, DEFAULTS };
export { symmetricEigen } from './linalg';
export type {
  VeritasConfig,
  ScoreOptions,
  VeritasComponents,
} from './engine';
export type { Vector, Matrix } from './linalg';

export default veritas;
