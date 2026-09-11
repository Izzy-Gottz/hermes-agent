/**
 * bridge.js is wired to the history store: the store is opened from
 * WHATSAPP_HISTORY_DB, and the socket asks the phone for its full history
 * when WHATSAPP_SYNC_FULL_HISTORY is on.
 *
 * history_store.test.mjs proves the store; nothing proved bridge.js uses it.
 * The makeWASocket settings are where the history store and the device name
 * (bridge.devicename.test.mjs) met in a merge, and a merge that keeps
 * `syncFullHistory: false` from one side silently asks the phone for the
 * recent slice only — every other suite still passes. So the socket line is
 * read here, the same way bridge.devicename.test.mjs reads `browser:`.
 *
 * Source checks, because importing bridge.js starts an HTTP server and a
 * Baileys socket at module load.
 */

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');

// -- the two settings are read from the environment ------------------------
{
  assert.match(src, /const HISTORY_DB = String\(process\.env\.WHATSAPP_HISTORY_DB \|\| ''\)\.trim\(\);/,
    'bridge.js does not read WHATSAPP_HISTORY_DB');
  assert.match(src, /const SYNC_FULL_HISTORY = envFlag\('WHATSAPP_SYNC_FULL_HISTORY'\);/,
    'bridge.js does not read WHATSAPP_SYNC_FULL_HISTORY');
  assert.match(src, /openHistoryStore\(HISTORY_DB\)/,
    'bridge.js does not open the store WHATSAPP_HISTORY_DB names');
  console.log('  ✓ the store and the full-history flag come from the environment');
}

// -- the socket is handed both, beside the device name ---------------------
{
  const socket = src.slice(src.indexOf('makeWASocket({'), src.indexOf('getMessage:'));
  assert.ok(socket.length > 0, 'makeWASocket settings not found in bridge.js');
  assert.match(socket, /syncFullHistory: SYNC_FULL_HISTORY,/,
    'makeWASocket does not take syncFullHistory from WHATSAPP_SYNC_FULL_HISTORY');
  assert.doesNotMatch(socket, /syncFullHistory: (false|true),/,
    'makeWASocket hardcodes syncFullHistory');
  assert.match(socket, /browser: browserDescription\(\),/,
    'makeWASocket lost the device name beside the history flag');
  console.log('  ✓ makeWASocket takes syncFullHistory from the flag, beside browserDescription()');
}

console.log('bridge.historywiring.test.mjs: all passed');
