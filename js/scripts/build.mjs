// Builds ESM, CJS and the minified browser (CDN) bundle with esbuild.
// Type declarations are emitted separately by `tsc` (see package.json).
import { build } from 'esbuild';
import { rmSync, statSync } from 'node:fs';

rmSync('dist', { recursive: true, force: true });

const common = { bundle: true, target: 'es2019', logLevel: 'info', legalComments: 'none' };

await build({ ...common, entryPoints: ['src/index.ts'], format: 'esm', outfile: 'dist/index.mjs', sourcemap: true });
await build({ ...common, entryPoints: ['src/index.ts'], format: 'cjs', outfile: 'dist/index.cjs', sourcemap: true });
await build({
  ...common,
  entryPoints: ['src/browser.ts'],
  format: 'iife',
  minify: true,
  outfile: 'dist/veritas.min.js',
});

const kb = (statSync('dist/veritas.min.js').size / 1024).toFixed(1);
console.log(`dist/veritas.min.js: ${kb} KB`);
