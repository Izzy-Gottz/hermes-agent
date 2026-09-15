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
  assert.match(src, /const SYNC_FULL_HISTORY = fullHistoryRequested\(process\.env\);/,
    'bridge.js does not read WHATSAPP_SYNC_FULL_HISTORY');
  assert.match(src, /openHistoryStore\(HISTORY_DB\)/,
    'bridge.js does not open the store WHATSAPP_HISTORY_DB names');
  console.log('  ✓ the store and the full-history flag come from the environment');
}

// -- the socket is handed both, beside the device name ---------------------
{
  const socket = src.slice(src.indexOf('makeWASocket({'), src.indexOf('getMessage:'));
  assert.ok(socket.length > 0, 'makeWASocket settings not found in bridge.js');
  assert.match(socket, /\.\.\.historySocketOptions\(SYNC_FULL_HISTORY\),/,
    'makeWASocket does not take its history settings from historySocketOptions(SYNC_FULL_HISTORY)');
  // Nothing after the spread may put either key back: a later
  // `syncFullHistory:` or `shouldSyncHistoryMessage:` in the same object
  // wins, and the second one is exactly the FULL-chunk drop this replaced.
  assert.doesNotMatch(socket, /^\s*syncFullHistory:/m,
    'makeWASocket sets syncFullHistory itself beside historySocketOptions');
  assert.doesNotMatch(socket, /^\s*shouldSyncHistoryMessage:/m,
    'makeWASocket sets shouldSyncHistoryMessage itself beside historySocketOptions');
  assert.match(socket, /browser: browserDescription\(\),/,
    'makeWASocket lost the device name beside the history flag');
  console.log('  ✓ makeWASocket takes its history settings from the flag, beside browserDescription()');
}

// -- what those settings DO, against Baileys' own defaults ------------------
// makeWASocket is `{ ...DEFAULT_CONNECTION_CONFIG, ...config }`
// (lib/Socket/index.js), so the effective config is built the same way here
// and asked the question Baileys asks before it keeps a history chunk
// (lib/Socket/chats.js: shouldSyncHistoryMessage(historyMsg) &&
// PROCESSABLE_HISTORY_TYPES.includes(syncType)).
{
  const { DEFAULT_CONNECTION_CONFIG, PROCESSABLE_HISTORY_TYPES, proto } = await import('@whiskeysockets/baileys');
  const { historySocketOptions } = await import('./bridge_helpers.js');
  const T = proto.HistorySync.HistorySyncType;
  const kept = (cfg, syncType) => cfg.shouldSyncHistoryMessage({ syncType }) && PROCESSABLE_HISTORY_TYPES.includes(syncType);

  // The premise. If Baileys ever stops dropping FULL by default this says
  // so, rather than the test below passing for a reason it does not name.
  assert.equal(DEFAULT_CONNECTION_CONFIG.shouldSyncHistoryMessage({ syncType: T.FULL }), false,
    "Baileys' default no longer drops FULL chunks — re-read Defaults/index.js");

  const on = { ...DEFAULT_CONNECTION_CONFIG, ...historySocketOptions(true) };
  assert.equal(on.syncFullHistory, true);
  assert.equal(kept(on, T.FULL), true, 'full history asked for, and the FULL chunks are still dropped');
  for (const t of PROCESSABLE_HISTORY_TYPES) {
    assert.equal(kept(on, t), true, `history type ${t} is dropped with full history on`);
  }

  const off = { ...DEFAULT_CONNECTION_CONFIG, ...historySocketOptions(false) };
  assert.equal(off.syncFullHistory, false);
  assert.equal('shouldSyncHistoryMessage' in historySocketOptions(false), false,
    "with full history off the bridge must leave Baileys' default alone");
  assert.equal(off.shouldSyncHistoryMessage, DEFAULT_CONNECTION_CONFIG.shouldSyncHistoryMessage);
  assert.equal(kept(off, T.RECENT), true, 'the recent slice is kept with full history off');
  assert.equal(kept(off, T.FULL), false);
  console.log("  ✓ full history on keeps every chunk, FULL included; off leaves Baileys' default");
}

// -- what the phone is told at pairing ------------------------------------
// The registration node Baileys builds from these settings, decoded: the
// name the phone lists, the kind of device, and whether full sync is asked.
{
  const { generateRegistrationNode, proto } = await import('@whiskeysockets/baileys');
  const { browserDescription, historySocketOptions, fullHistoryRequested } = await import('./bridge_helpers.js');
  const creds = {
    registrationId: 1,
    signedIdentityKey: { public: new Uint8Array(32) },
    signedPreKey: { keyId: 1, keyPair: { public: new Uint8Array(32) }, signature: new Uint8Array(64) },
  };
  const props = (env) => {
    const payload = generateRegistrationNode(creds, {
      version: [2, 3000, 1],
      browser: browserDescription(env),
      ...historySocketOptions(fullHistoryRequested(env)),
    });
    return proto.DeviceProps.decode(payload.devicePairingData.deviceProps);
  };
  const on = props({ WHATSAPP_DEVICE_NAME: 'Moe', WHATSAPP_SYNC_FULL_HISTORY: 'true' });
  assert.equal(on.os, 'Moe', 'the phone must still list the link as Moe');
  assert.equal(on.requireFullSync, true);
  assert.equal(on.platformType, proto.DeviceProps.PlatformType.DESKTOP,
    "full history is Baileys' Desktop recipe (README: Browsers.macOS('Desktop'))");
  const off = props({ WHATSAPP_DEVICE_NAME: 'Moe' });
  assert.equal(off.os, 'Moe');
  assert.ok(!off.requireFullSync);
  assert.equal(off.platformType, proto.DeviceProps.PlatformType.CHROME, 'full history off is unchanged');
  console.log('  ✓ at pairing: named Moe either way; Desktop and requireFullSync only with full history on');
}

console.log('bridge.historywiring.test.mjs: all passed');
