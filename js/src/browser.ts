// Entry point for the standalone CDN bundle (dist/veritas.min.js).
// Attaches `veritas` and `VeritasEngine` to `window` for <script> tag usage.
import veritas, { VeritasEngine } from './index';

const root = (typeof window !== 'undefined' ? window : globalThis) as unknown as Record<string, unknown>;
root.veritas = veritas;
root.VeritasEngine = VeritasEngine;
