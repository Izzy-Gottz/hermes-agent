/**
 * A Node without node:sqlite loses the history store and keeps the bridge.
 *
 * node:sqlite is unflagged from Node 22.13 / 23.4; 22.5–22.12 hide it behind
 * --experimental-sqlite and older Nodes do not have it. history_store.js
 * used to import it at the top, and bridge.js imports history_store.js
 * unconditionally — so on such a Node the whole bridge failed at load, the
 * bot with it, even with WHATSAPP_HISTORY_DB unset.
 *
 * The missing module is simulated with a module-customisation hook that
 * refuses 'node:sqlite' the way an old Node does (ERR_UNKNOWN_BUILTIN_MODULE).
 * The second half runs the REAL bridge.js under that hook, with Baileys
 * replaced by a stub that never opens a socket (no network, no phone), and
 * requires its HTTP server to answer /health with history: null.
 */

import { strict as assert } from 'node:assert';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = mkdtempSync(path.join(tmpdir(), 'wa-nosqlite-'));

const stubBaileys = path.join(dir, 'baileys-stub.mjs');
writeFileSync(stubBaileys, `
const noop = () => {};
export const DisconnectReason = { loggedOut: 401 };
export const makeWASocket = () => ({ ev: { on: noop }, ws: { on: noop, close: noop }, end: noop, user: null });
export const useMultiFileAuthState = async () => ({ state: { creds: {}, keys: {} }, saveCreds: noop });
export const fetchLatestBaileysVersion = async () => ({ version: [2, 3000, 1] });
export const downloadMediaMessage = noop, downloadContentFromMessage = noop,
  getAggregateVotesInPollMessage = noop, decryptPollVote = noop, getKeyAuthor = noop,
  jidNormalizedUser = (j) => j;
`);
const hooks = path.join(dir, 'hooks.mjs');
writeFileSync(hooks, `
let stub = null;
export async function initialize(data) { stub = data?.stub || null; }
export async function resolve(specifier, context, next) {
  if (specifier === 'node:sqlite') {
    const err = new Error('No such built-in module: node:sqlite');
    err.code = 'ERR_UNKNOWN_BUILTIN_MODULE';
    throw err;
  }
  if (stub && specifier === '@whiskeysockets/baileys') return { url: stub, shortCircuit: true };
  return next(specifier, context);
}
`);
const register = (withStub) => {
  const f = path.join(dir, withStub ? 'register-stub.mjs' : 'register.mjs');
  writeFileSync(f, `import { register } from 'node:module';
register(${JSON.stringify(pathToFileURL(hooks).href)}, { data: { stub: ${withStub ? JSON.stringify(pathToFileURL(stubBaileys).href) : 'null'} } });
`);
  return f;
};

try {
  // -- the hook really takes node:sqlite away (else this proves nothing) --
  {
    const r = spawnSync(process.execPath, ['--no-warnings', '--import', register(false), '--input-type=module', '-e',
      "try { await import('node:sqlite'); console.log('HAS'); } catch (e) { console.log('NONE ' + e.code); }"],
    { cwd: here, encoding: 'utf8' });
    assert.match(r.stdout, /^NONE ERR_UNKNOWN_BUILTIN_MODULE/m, `the hook did not hide node:sqlite: ${r.stdout}${r.stderr}`);
    console.log('  ✓ under the hook, node:sqlite is missing the way an old Node has it missing');
  }

  // -- history_store.js loads; opening a store says why it cannot ----------
  {
    const db = path.join(dir, 'store', 'messages.db');
    const r = spawnSync(process.execPath, ['--no-warnings', '--import', register(false), '--input-type=module', '-e', `
      const hs = await import(${JSON.stringify(pathToFileURL(path.join(here, 'history_store.js')).href)});
      console.log('LOADED');
      try { await hs.openHistoryStore(${JSON.stringify(db)}); console.log('OPENED'); }
      catch (e) { console.log('REFUSED ' + e.message); }`],
    { cwd: here, encoding: 'utf8' });
    assert.match(r.stdout, /^LOADED$/m, `history_store.js failed to load without node:sqlite: ${r.stderr.split('\n').slice(0, 4).join(' | ')}`);
    assert.match(r.stdout, /^REFUSED .*node:sqlite/m, `openHistoryStore did not refuse with a reason: ${r.stdout}${r.stderr}`);
    console.log('  ✓ history_store.js loads without node:sqlite; openHistoryStore refuses and names it');
  }

  // -- the real bridge.js starts and answers, with no store ------------------
  {
    const port = await new Promise((res) => { const s = createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
    const child = spawn(process.execPath, ['--no-warnings', '--import', register(true), path.join(here, 'bridge.js'),
      '--port', String(port), '--session', path.join(dir, 'session')], {
      cwd: here,
      env: { ...process.env, WHATSAPP_HISTORY_DB: path.join(dir, 'store', 'messages.db'), WHATSAPP_SYNC_FULL_HISTORY: 'true', WHATSAPP_MODE: 'self-chat' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    child.stdout.on('data', (b) => { out += b; });
    child.stderr.on('data', (b) => { out += b; });
    let health = null;
    try {
      for (let i = 0; i < 100 && health === null && child.exitCode === null; i += 1) {
        await new Promise((r) => setTimeout(r, 100));
        try {
          const r = await fetch(`http://127.0.0.1:${port}/health`);
          health = await r.json();
        } catch { /* not up yet */ }
      }
    } finally {
      child.kill('SIGKILL');
    }
    assert.ok(health, `bridge.js did not come up without node:sqlite (exit ${child.exitCode}):\n${out.split('\n').slice(0, 8).join('\n')}`);
    assert.equal(health.history, null, `/health should report history: null, got ${JSON.stringify(health.history)}`);
    assert.match(out, /history store unavailable: .*node:sqlite/, `the bridge did not say why it keeps no history:\n${out}`);
    console.log('  ✓ bridge.js serves /health with history: null and says node:sqlite is missing');
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log('bridge.nosqlite.test.mjs: all passed');
