/**
 * Unit tests for clearing the Baileys auth state after the phone unlinks us.
 *
 * Regression test for the logout loop: on DisconnectReason.loggedOut (401)
 * the bridge exited 1 and left every credential on disk. The gateway's
 * preflight (plugins/platforms/whatsapp/adapter.py) decides "is this paired?"
 * by asking whether creds.json exists, so a revoked session still read as
 * paired — the adapter restarted the bridge, it was logged out again, and the
 * person saw "bridge process exited unexpectedly (code 1)" for as long as
 * they left it running. The only escape was deleting files by hand, which is
 * exactly what the old console message told a human to do and what nothing
 * ever did on its own.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and a Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, mkdirSync, writeFileSync, readdirSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { clearAuthState } from './bridge_helpers.js';

const seed = (files) => {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-logout-'));
  for (const [name, body] of Object.entries(files)) {
    writeFileSync(path.join(dir, name), body);
  }
  return dir;
};

// -- the whole auth folder goes, whatever Baileys chose to call its keys ---

{
  const dir = seed({
    'creds.json': '{}',
    'app-state-sync-key-AAAAA.json': '{}',
    'pre-key-1.json': '{}',
    'session-15551234567.0.json': '{}',
    'sender-key-x@g.us--y.json': '{}',
    'lid-mapping-15551234567.json': '"123"',
    // A key type no version of Baileys has shipped yet. A fix that listed
    // prefixes would leave this one behind; deleting by extension does not.
    'some-future-key-type-9.json': '{}',
  });

  const removed = clearAuthState(dir);

  assert.equal(readdirSync(dir).length, 0, 'auth state survived the logout');
  assert.equal(removed.length, 7);
  assert.ok(removed.includes('creds.json'), 'creds.json is the file the preflight reads');
  assert.ok(
    removed.includes('some-future-key-type-9.json'),
    'an unknown key type must be cleared too, or the next Baileys walks past the fix',
  );
}

// -- and nothing else does -------------------------------------------------

// The message store is a SIBLING of the session directory
// (whatsapp/store/messages.db, not whatsapp/session/), but if someone ever
// points WHATSAPP_HISTORY_DB inside it, a logout must not eat their history.
{
  const dir = seed({
    'creds.json': '{}',
    'messages.db': 'SQLite format 3\0',
    'notes.txt': 'keep me',
  });
  mkdirSync(path.join(dir, 'subdir'));

  const removed = clearAuthState(dir);

  assert.deepEqual(removed, ['creds.json']);
  assert.ok(existsSync(path.join(dir, 'messages.db')), 'the message store was deleted');
  assert.ok(existsSync(path.join(dir, 'notes.txt')), 'a non-json file was deleted');
  assert.ok(existsSync(path.join(dir, 'subdir')), 'a subdirectory was deleted');
}

// -- it never throws on the way out of an exiting process ------------------

// A directory that is not there at all.
{
  const removed = clearAuthState(path.join(tmpdir(), 'wa-logout-does-not-exist-' + Date.now()));
  assert.deepEqual(removed, [], 'a missing session directory must be survivable');
}

// One undeletable file must not stop the others — creds.json still goes,
// because that is the one the preflight reads.
{
  const dir = seed({ 'creds.json': '{}', 'pre-key-1.json': '{}' });
  const real = readdirSync(dir);
  const fs = {
    readdirSync: () => real,
    unlinkSync: (p) => {
      if (p.endsWith('pre-key-1.json')) throw new Error('EPERM');
    },
  };

  const removed = clearAuthState(dir, fs);

  assert.deepEqual(removed, ['creds.json']);
}

// -- bridge.js actually calls it, on the loggedOut branch and nowhere else --

// The helper working proves nothing if the logout path never reaches it, and
// bridge.js cannot be imported here to check by running it.
{
  const src = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');

  assert.match(src, /clearAuthState,/, 'bridge.js does not import clearAuthState');

  const logout = src.slice(
    src.indexOf('if (reason === DisconnectReason.loggedOut)'),
    src.indexOf('} else {', src.indexOf('if (reason === DisconnectReason.loggedOut)')),
  );
  assert.ok(logout.length > 0, 'the loggedOut branch moved; this test cannot see it');
  assert.match(logout, /clearAuthState\(SESSION_DIR\)/,
    'the loggedOut branch does not clear the revoked credentials');
  assert.ok(
    logout.indexOf('clearAuthState(SESSION_DIR)') < logout.indexOf('process.exit'),
    'credentials are cleared after the process exits, which is never',
  );
  assert.ok(
    !/Delete session and restart/.test(src),
    'bridge.js still tells a human to delete the session by hand',
  );
}

console.log('bridge.logout.test.mjs: all assertions passed');
