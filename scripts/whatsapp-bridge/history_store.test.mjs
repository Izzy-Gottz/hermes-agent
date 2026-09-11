/**
 * Tests for history_store.js — the SQLite mirror the bridge keeps of every
 * message the linked device sees.
 *
 * No socket, no Baileys, no network: the pure helpers take plain objects
 * shaped like Baileys' (keys, Long timestamps, LID jids with their *Alt
 * phone-number twins), and the store is opened on a file in a temp dir and
 * read back through the same node:sqlite the bridge uses.
 *
 * The shapes here are copied from what Baileys 7.0.0-rc13 hands the bridge,
 * not invented — a store that agrees with its own test and disagrees with
 * the socket is the trap this file exists to avoid.
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  addLidMappings,
  chatRowsFromHistory,
  contactRows,
  historyTimestamp,
  jidUser,
  openHistoryStore,
  rowFromMessage,
  timestampSeconds,
  wholeNumber,
} from './history_store.js';

// ------------------------------------------------------------------
// 1. Timestamps: number, Long-like {low, high}, {toNumber}, string, ms.
// ------------------------------------------------------------------
{
  assert.equal(timestampSeconds(1700000000), 1700000000);
  assert.equal(timestampSeconds('1700000000'), 1700000000);
  assert.equal(timestampSeconds({ low: 1700000000, high: 0, unsigned: false }), 1700000000);
  assert.equal(timestampSeconds({ toNumber: () => 1700000000 }), 1700000000);
  // Milliseconds, should anything upstream ever hand those.
  assert.equal(timestampSeconds(1700000000123), 1700000000);
  assert.equal(timestampSeconds(0), null);
  assert.equal(timestampSeconds(undefined), null);
  assert.equal(timestampSeconds('soon'), null);
  // The format a reader compares as text against sqlite's own datetime('now').
  assert.equal(historyTimestamp(1700000000), '2023-11-14 22:13:20');
  assert.equal(historyTimestamp(null), null);
  // Sizes are not timestamps: 0 is a size, and nothing is "milliseconds".
  assert.equal(wholeNumber(0), 0);
  assert.equal(wholeNumber(123456789012), 123456789012);
  assert.equal(wholeNumber({ low: 5, high: 0 }), 5);
  assert.equal(wholeNumber(-1), null);
  assert.equal(jidUser('447700900001:12@s.whatsapp.net'), '447700900001');
  assert.equal(jidUser('12345@g.us'), '12345');
}

// ------------------------------------------------------------------
// 2. rowFromMessage: text, media, the things that are not messages.
// ------------------------------------------------------------------
{
  const text = rowFromMessage({
    key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: false, id: 'A1' },
    messageTimestamp: { low: 1700000000, high: 0 },
    pushName: 'Ada',
    message: { conversation: 'hello' },
  });
  assert.deepEqual(
    { id: text.id, chat: text.chat_jid, sender: text.sender, content: text.content, ts: text.timestamp, me: text.is_from_me, media: text.media_type, push: text.push_name },
    { id: 'A1', chat: '447700900001@s.whatsapp.net', sender: '447700900001', content: 'hello', ts: '2023-11-14 22:13:20', me: 0, media: '', push: 'Ada' },
  );

  // Mine: sender is still the chat's number (a one-to-one chat), and MY
  // push name is not a name for the other person.
  const mine = rowFromMessage({
    key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: true, id: 'A2' },
    messageTimestamp: 1700000001,
    pushName: 'Me',
    message: { extendedTextMessage: { text: 'hi back' } },
  });
  assert.equal(mine.is_from_me, 1);
  assert.equal(mine.content, 'hi back');
  assert.equal(mine.push_name, '');

  // A group: sender is the participant, the chat is the group.
  const group = rowFromMessage({
    key: { remoteJid: '120363@g.us', fromMe: false, id: 'G1', participant: '447700900002@s.whatsapp.net' },
    messageTimestamp: 1700000002,
    pushName: 'Bob',
    message: { conversation: 'in the group' },
  });
  assert.equal(group.chat_jid, '120363@g.us');
  assert.equal(group.sender, '447700900002');
  assert.equal(group.is_group, true);
  assert.equal(group.push_name, '', 'a push name in a group is not the chat name');

  // A voice note: media_type audio, the download reference kept.
  const voice = rowFromMessage({
    key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: false, id: 'V1' },
    messageTimestamp: 1700000003,
    message: { audioMessage: { ptt: true, mimetype: 'audio/ogg; codecs=opus', url: 'https://mmg.whatsapp.net/v/x.enc', directPath: '/v/x.enc', mediaKey: new Uint8Array([1, 2, 3]), fileLength: { low: 4321, high: 0 } } },
  });
  assert.equal(voice.media_type, 'audio');
  assert.equal(voice.content, '');
  assert.equal(voice.url, 'https://mmg.whatsapp.net/v/x.enc');
  assert.equal(voice.direct_path, '/v/x.enc');
  assert.deepEqual(Array.from(voice.media_key), [1, 2, 3]);
  assert.equal(voice.file_length, 4321);
  assert.equal(voice.mimetype, 'audio/ogg; codecs=opus');

  // A photo with a caption: the caption is the content.
  const photo = rowFromMessage({
    key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: false, id: 'P1' },
    messageTimestamp: 1700000004,
    message: { imageMessage: { caption: 'look', mimetype: 'image/jpeg', mediaKey: 'AQID' } },
  });
  assert.equal(photo.media_type, 'image');
  assert.equal(photo.content, 'look');
  assert.deepEqual(Array.from(photo.media_key), [1, 2, 3], 'a base64 media key is decoded');

  // Wrapped: an ephemeral envelope is peeled the way the bridge peels it.
  const wrapped = rowFromMessage({
    key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: false, id: 'E1' },
    messageTimestamp: 1700000005,
    message: { ephemeralMessage: { message: { conversation: 'disappearing' } } },
  });
  assert.equal(wrapped.content, 'disappearing');

  // Not messages: a reaction, a protocol frame, a status post, a receipt.
  assert.equal(rowFromMessage({ key: { remoteJid: '447700900001@s.whatsapp.net', id: 'R1' }, messageTimestamp: 1, message: { reactionMessage: { text: '👍' } } }), null);
  assert.equal(rowFromMessage({ key: { remoteJid: '447700900001@s.whatsapp.net', id: 'R2' }, messageTimestamp: 1, message: { protocolMessage: { type: 0 } } }), null);
  assert.equal(rowFromMessage({ key: { remoteJid: 'status@broadcast', id: 'S1' }, messageTimestamp: 1, message: { conversation: 'my status' } }), null);
  assert.equal(rowFromMessage({ key: { remoteJid: '447700900001@s.whatsapp.net', id: 'X' }, messageTimestamp: 1, message: null }), null);
  assert.equal(rowFromMessage({ key: { id: 'X' }, message: { conversation: 'no chat' } }), null);
  assert.equal(rowFromMessage(null), null);
}

// ------------------------------------------------------------------
// 3. LIDs: the phone number is stored when it is knowable, three ways.
// ------------------------------------------------------------------
{
  // (a) Baileys 7 carries the phone jid beside the LID key.
  const alt = rowFromMessage({
    key: { remoteJid: '9988776655@lid', remoteJidAlt: '447700900001@s.whatsapp.net', fromMe: false, id: 'L1' },
    messageTimestamp: 1700000010,
    message: { conversation: 'from a lid' },
  });
  assert.equal(alt.chat_jid, '447700900001@s.whatsapp.net');
  assert.equal(alt.sender, '447700900001');

  // (b) The session's lid-mapping files, as bridge.js reads them.
  const mapped = rowFromMessage({
    key: { remoteJid: '9988776655@lid', fromMe: false, id: 'L2' },
    messageTimestamp: 1700000011,
    message: { conversation: 'mapped' },
  }, { lidToPhone: { '9988776655': '447700900001' } });
  assert.equal(mapped.chat_jid, '447700900001@s.whatsapp.net');

  // (c) The history sync's own lidPnMappings, folded into that map.
  const map = addLidMappings({}, [{ lid: '9988776655@lid', pn: '447700900001@s.whatsapp.net' }, { lid: 'bad' }, null]);
  assert.deepEqual(map, { '9988776655': '447700900001' });

  // Unknowable: the LID is stored as it is, never invented.
  const raw = rowFromMessage({
    key: { remoteJid: '1122334455@lid', fromMe: false, id: 'L3' },
    messageTimestamp: 1700000012,
    message: { conversation: 'unknown lid' },
  }, { lidToPhone: {} });
  assert.equal(raw.chat_jid, '1122334455@lid');
  assert.equal(raw.sender, '1122334455');

  // A group participant on a LID, resolved through participantAlt.
  const gp = rowFromMessage({
    key: { remoteJid: '120363@g.us', fromMe: false, id: 'L4', participant: '9988776655@lid', participantAlt: '447700900001@s.whatsapp.net' },
    messageTimestamp: 1700000013,
    message: { conversation: 'lid in a group' },
  });
  assert.equal(gp.sender, '447700900001');
}

// ------------------------------------------------------------------
// 4. Chats and contacts from a history sync.
// ------------------------------------------------------------------
{
  const chats = chatRowsFromHistory([
    { id: '447700900001@s.whatsapp.net', name: 'Ada', conversationTimestamp: 1700000020 },
    { id: '120363@g.us', name: 'Family', conversationTimestamp: { low: 1700000021, high: 0 } },
    // A LID chat WITH its phone jid: the phone jid is the row. This was the
    // precedence bug — `a || b && c ? d : e` handed such a chat its lidJid.
    { id: '9988776655@lid', pnJid: '447700900003@s.whatsapp.net', lidJid: '9988776655@lid', name: 'Cy', conversationTimestamp: 1700000022 },
    { id: 'status@broadcast', name: 'Status' },
    { name: 'no id' },
  ]);
  assert.deepEqual(chats, [
    { jid: '447700900001@s.whatsapp.net', name: 'Ada', last_message_time: '2023-11-14 22:13:40' },
    { jid: '120363@g.us', name: 'Family', last_message_time: '2023-11-14 22:13:41' },
    { jid: '447700900003@s.whatsapp.net', name: 'Cy', last_message_time: '2023-11-14 22:13:42' },
  ]);

  const contacts = contactRows([
    { id: '447700900001@s.whatsapp.net', name: 'Ada Lovelace', notify: 'ada' },   // saved name wins
    { id: '447700900002@s.whatsapp.net', notify: 'bob-on-wa' },                     // else their own
    { id: '9988776655@lid', phoneNumber: '447700900003@s.whatsapp.net', name: 'Cy' }, // by phone
    { id: '447700900004@s.whatsapp.net' },                                          // nameless: skipped
  ]);
  assert.deepEqual(contacts, [
    { jid: '447700900001@s.whatsapp.net', name: 'Ada Lovelace' },
    { jid: '447700900002@s.whatsapp.net', name: 'bob-on-wa' },
    { jid: '447700900003@s.whatsapp.net', name: 'Cy' },
  ]);
}

// ------------------------------------------------------------------
// 5. The store on disk: what a reader with sqlite3 sees.
// ------------------------------------------------------------------
{
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-history-'));
  const file = path.join(dir, 'store', 'messages.db');
  try {
    const store = openHistoryStore(file);
    assert.ok(existsSync(file), 'the file and its directory are created');
    assert.deepEqual(store.counts(), { messages: 0, chats: 0 });

    const m = (id, jid, text, ts, fromMe = false, extra = {}) => ({
      key: { remoteJid: jid, fromMe, id },
      messageTimestamp: ts,
      message: { conversation: text },
      ...extra,
    });
    // Two messages in one chat, a third in another, and a duplicate.
    assert.equal(store.recordMessages([
      m('A1', '447700900001@s.whatsapp.net', 'one', 1700000000, false, { pushName: 'Ada' }),
      m('A2', '447700900001@s.whatsapp.net', 'two', 1700000100, true),
      m('B1', '447700900002@s.whatsapp.net', 'other', 1700000050, false),
      m('A1', '447700900001@s.whatsapp.net', 'one again', 1700000000),
    ]), 3, 'a message seen twice is stored once');
    assert.deepEqual(store.counts(), { messages: 3, chats: 2 });

    // Read back the way whatsapp.sh does: sqlite, joined on chats.
    const ro = new DatabaseSync(file, { readOnly: true });
    const rows = ro.prepare(`
      SELECT m.id, COALESCE(NULLIF(c.name,''), m.chat_jid) AS who, m.content, m.timestamp, m.is_from_me, c.last_message_time
      FROM messages m LEFT JOIN chats c ON c.jid = m.chat_jid ORDER BY m.timestamp`).all();
    assert.deepEqual(rows.map(r => [r.id, r.who, r.content, r.timestamp, r.is_from_me, r.last_message_time]), [
      ['A1', 'Ada', 'one', '2023-11-14 22:13:20', 0, '2023-11-14 22:15:00'],
      ['B1', '447700900002@s.whatsapp.net', 'other', '2023-11-14 22:14:10', 0, '2023-11-14 22:14:10'],
      ['A2', 'Ada', 'two', '2023-11-14 22:15:00', 1, '2023-11-14 22:15:00'],
    ]);
    // The push name named the chat only because nothing better was known…
    // …and a saved contact then replaces it, and names the other chat too.
    assert.equal(store.recordContacts([
      { id: '447700900001@s.whatsapp.net', name: 'Ada Lovelace' },
      { id: '447700900002@s.whatsapp.net', notify: 'bob' },
    ]), 2);
    const names = ro.prepare('SELECT jid, name FROM chats ORDER BY jid').all().map(r => [r.jid, r.name]);
    assert.deepEqual(names, [['447700900001@s.whatsapp.net', 'Ada Lovelace'], ['447700900002@s.whatsapp.net', 'bob']]);

    // A later message from Ada does not un-name the chat with her push name…
    store.recordMessages([m('A3', '447700900001@s.whatsapp.net', 'three', 1700000200, false, { pushName: 'ada-phone' })]);
    assert.equal(ro.prepare('SELECT name FROM chats WHERE jid = ?').get('447700900001@s.whatsapp.net').name, 'Ada Lovelace');
    // …and an OLDER message (history arriving late) never moves last_message_time backwards.
    store.recordMessages([m('A0', '447700900001@s.whatsapp.net', 'zero', 1600000000)]);
    assert.equal(ro.prepare('SELECT last_message_time FROM chats WHERE jid = ?').get('447700900001@s.whatsapp.net').last_message_time, '2023-11-14 22:16:40');

    // Chats from a history sync: a name arrives for a chat with none, an
    // empty name never blanks one that exists.
    assert.equal(store.recordChats([
      { id: '120363@g.us', name: 'Family', conversationTimestamp: 1700000300 },
      { id: '447700900001@s.whatsapp.net', name: '', conversationTimestamp: 1500000000 },
    ]), 2);
    assert.equal(ro.prepare('SELECT name FROM chats WHERE jid = ?').get('120363@g.us').name, 'Family');
    assert.equal(ro.prepare('SELECT name FROM chats WHERE jid = ?').get('447700900001@s.whatsapp.net').name, 'Ada Lovelace');
    // A group's subject from groups.upsert.
    store.nameChat('120363@g.us', 'Family (2026)');
    assert.equal(ro.prepare('SELECT name FROM chats WHERE jid = ?').get('120363@g.us').name, 'Family (2026)');
    store.nameChat('120363@g.us', '');
    assert.equal(ro.prepare('SELECT name FROM chats WHERE jid = ?').get('120363@g.us').name, 'Family (2026)', 'an empty subject changes nothing');

    // Media: what a download needs comes back whole; a text message has none.
    store.recordMessages([{
      key: { remoteJid: '447700900001@s.whatsapp.net', fromMe: false, id: 'V1' },
      messageTimestamp: 1700000400,
      message: { audioMessage: { ptt: true, mimetype: 'audio/ogg; codecs=opus', url: 'https://mmg/x.enc', directPath: '/x.enc', mediaKey: new Uint8Array([9, 8]), fileLength: 77 } },
    }]);
    const ref = store.mediaRef('V1', '447700900001@s.whatsapp.net');
    assert.equal(ref.type, 'audio');
    assert.equal(ref.url, 'https://mmg/x.enc');
    assert.equal(ref.directPath, '/x.enc');
    assert.deepEqual(Array.from(ref.mediaKey), [9, 8]);
    assert.equal(ref.fileLength, 77);
    assert.equal(store.mediaRef('A1', '447700900001@s.whatsapp.net'), null);
    assert.equal(store.mediaRef('nope', 'nowhere'), null);
    // The voice query whatsapp.sh runs.
    const voices = ro.prepare("SELECT id FROM messages WHERE media_type = 'audio' ORDER BY timestamp DESC").all();
    assert.deepEqual(voices.map(v => v.id), ['V1']);

    // The timestamp column compares as text against sqlite's own clock —
    // the reason the format is fixed.
    const recent = ro.prepare("SELECT count(*) AS n FROM messages WHERE timestamp > datetime('now', '-100 years')").get().n;
    assert.equal(recent, 6);
    const none = ro.prepare("SELECT count(*) AS n FROM messages WHERE timestamp > datetime('now', '+1 day')").get().n;
    assert.equal(none, 0);

    // An empty batch is a no-op, and a batch of non-messages too.
    assert.equal(store.recordMessages([]), 0);
    assert.equal(store.recordMessages([{ key: { remoteJid: '447700900001@s.whatsapp.net', id: 'R' }, messageTimestamp: 1, message: { reactionMessage: {} } }]), 0);
    assert.equal(store.recordChats([]), 0);
    assert.equal(store.recordContacts([{ id: 'x@s.whatsapp.net' }]), 0);

    ro.close();
    store.close();
    // Reopening keeps everything: the schema is CREATE IF NOT EXISTS.
    const again = openHistoryStore(file);
    assert.deepEqual(again.counts(), { messages: 6, chats: 3 });
    again.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

console.log('history_store.test.mjs: all assertions passed');
