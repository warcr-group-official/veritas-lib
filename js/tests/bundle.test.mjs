// Checks the shipped artifacts: CommonJS build and the minified CDN bundle.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { existsSync, readFileSync, statSync } from 'node:fs';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const cjsPath = new URL('../dist/index.cjs', import.meta.url);
const bundlePath = new URL('../dist/veritas.min.js', import.meta.url);

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} !≈ ${b}`);

test('CommonJS build exposes veritas and VeritasEngine', { skip: !existsSync(cjsPath) }, () => {
  const lib = require('../dist/index.cjs');
  close(lib.veritas([1, 2, 3]), 14);
  close(new lib.VeritasEngine({ alpha: 2 }).score([1, 2, 3]), 28);
});

test('CDN bundle attaches veritas and VeritasEngine to window', { skip: !existsSync(bundlePath) }, () => {
  const sandbox = {};
  sandbox.window = sandbox; // minimal browser-like global
  vm.createContext(sandbox);
  vm.runInContext(readFileSync(bundlePath, 'utf8'), sandbox);

  assert.equal(typeof sandbox.veritas, 'function');
  assert.equal(typeof sandbox.VeritasEngine, 'function');
  close(sandbox.veritas([1, 2, 3]), 14);
  close(new sandbox.VeritasEngine({ alpha: 2 }).score([1, 2, 3]), 28);
  assert.ok(statSync(bundlePath).size < 20 * 1024, 'bundle should stay lightweight (<20 KB)');
});
