/**
 * A local history of every message the linked WhatsApp device sees, in one
 * SQLite file, kept by the bridge itself.
 *
 * Why the bridge keeps it. The bridge is the ONE linked device for a
 * person's phone (every extra linked device is its own expiry and its own
 * share of the ban risk), and Baileys already hands it every chat, contact
 * and message — the initial history sync at link time and every upsert
 * after. Until this module the bridge forwarded the few messages the
 * gateway's policy admitted and dropped the rest on the floor, so a reader
 * that wanted "what did Dad say last week" had nothing to read short of a
 * SECOND device with its own store. One device, one store, read by anyone
 * with sqlite3.
 *
 * Off unless WHATSAPP_HISTORY_DB names a file. Nothing here changes what
 * the gateway receives: the store is written BEFORE the bot's own policy
 * (mode, allowlist, echo) decides whether a message is forwarded, and that
 * policy is untouched.
 *
 * The schema is the one lharries/whatsapp-mcp's Go bridge wrote, plus four
 * columns (direct_path, mimetype, and a contacts table) — a reader written
 * against that store keeps working against this one:
 *
 *   chats    (jid PK, name, last_message_time)
 *   messages (id, chat_jid) PK, sender, content, timestamp, is_from_me,
 *            media_type, filename, url, media_key, file_sha256,
 *            file_enc_sha256, file_length, direct_path, mimetype
 *   contacts (jid PK, name)
 *
 * Timestamps are UTC text, "YYYY-MM-DD HH:MM:SS" — what sqlite's own
 * datetime('now') produces, so a reader's `timestamp > datetime('now',
 * '-24 hours')` compares correctly as text and datetime(timestamp,
 * 'localtime') renders it. The Go store wrote Go's time format with a
 * zone suffix, which sqlite parsed but compared wrongly across the zone.
 *
 * node:sqlite, not a native addon: the Node that ships beside this file is
 * signed with a hardened runtime, and a fresh .node binary from npm would
 * be refused by it ("different Team IDs"). Built-in, nothing to sign.
 *
 * Pure helpers (rowFromMessage, historyTimestamp, chatRowsFromHistory,
 * contactRows) take plain objects and touch nothing, so they are tested
 * without Baileys or a socket.
 */

import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { getMessageContent } from './bridge_helpers.js';

export const SCHEMA = `
CREATE TABLE IF NOT EXISTS chats (
  jid TEXT PRIMARY KEY,
  name TEXT,
  last_message_time TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT,
  chat_jid TEXT,
  sender TEXT,
  content TEXT,
  timestamp TIMESTAMP,
  is_from_me BOOLEAN,
  media_type TEXT,
  filename TEXT,
  url TEXT,
  media_key BLOB,
  file_sha256 BLOB,
  file_enc_sha256 BLOB,
  file_length INTEGER,
  direct_path TEXT,
  mimetype TEXT,
  PRIMARY KEY (id, chat_jid),
  FOREIGN KEY (chat_jid) REFERENCES chats(jid)
);
CREATE INDEX IF NOT EXISTS messages_by_time ON messages(timestamp);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_jid, timestamp);
CREATE TABLE IF NOT EXISTS contacts (
  jid TEXT PRIMARY KEY,
  name TEXT
);
`;

// Baileys hands timestamps as a number, a Long ({low, high}), or a decimal
// string, depending on the path a message took. Seconds since the epoch.
export function timestampSeconds(ts) {
  if (ts === null || ts === undefined) return null;
  let n;
  if (typeof ts === 'object') {
    if (typeof ts.toNumber === 'function') n = ts.toNumber();
    else if ('low' in ts) n = (Number(ts.high) || 0) * 4294967296 + (Number(ts.low) >>> 0);
    else n = Number(ts);
  } else {
    n = Number(ts);
  }
  if (!Number.isFinite(n) || n <= 0) return null;
  // Milliseconds, if something upstream ever hands those.
  if (n > 1e11) n = Math.floor(n / 1000);
  return Math.floor(n);
}

/** A plain non-negative integer from a number, Long or decimal string, or null. */
export function wholeNumber(v) {
  if (v === null || v === undefined) return null;
  let n;
  if (typeof v === 'object') {
    if (typeof v.toNumber === 'function') n = v.toNumber();
    else if ('low' in v) n = (Number(v.high) || 0) * 4294967296 + (Number(v.low) >>> 0);
    else n = Number(v);
  } else {
    n = Number(v);
  }
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null;
}

/** "YYYY-MM-DD HH:MM:SS" in UTC, or null. */
export function historyTimestamp(ts) {
  const s = timestampSeconds(ts);
  if (s === null) return null;
  return new Date(s * 1000).toISOString().slice(0, 19).replace('T', ' ');
}

/** The user part of a jid: "4477…:12@s.whatsapp.net" → "4477…". */
export function jidUser(jid) {
  return String(jid || '').replace(/@.*$/, '').replace(/:.*$/, '');
}

function isLid(jid) {
  return String(jid || '').endsWith('@lid');
}

// A phone-number jid for a chat or sender when one is knowable: the
// *Alt fields Baileys 7 carries beside a LID key, then the session's
// lid-mapping files. Names come from Contacts by number, and a message is
// sent to a number, so a LID that can be resolved is stored resolved.
function phoneJid(jid, alt, lidToPhone) {
  if (!jid) return '';
  if (!isLid(jid)) return jid;
  if (alt && !isLid(alt)) return alt;
  const phone = lidToPhone && lidToPhone[jidUser(jid)];
  return phone ? `${phone}@s.whatsapp.net` : jid;
}

const MEDIA = [
  ['imageMessage', 'image'],
  ['videoMessage', 'video'],
  ['audioMessage', 'audio'],
  ['pttMessage', 'audio'],
  ['documentMessage', 'document'],
  ['stickerMessage', 'sticker'],
];

function bytes(v) {
  if (v === null || v === undefined) return null;
  if (v instanceof Uint8Array) return v;
  if (typeof v === 'string') return Buffer.from(v, 'base64');
  if (v && typeof v === 'object' && v.type === 'Buffer' && Array.isArray(v.data)) return Uint8Array.from(v.data);
  return null;
}

/**
 * Fold a history sync's `lidPnMappings` ([{lid, pn}]) into a LID-user →
 * phone map of the shape bridge.js builds from the session's lid-mapping
 * files. Returns the same map, mutated.
 */
export function addLidMappings(lidToPhone, mappings) {
  const map = lidToPhone || {};
  for (const m of mappings || []) {
    const lid = jidUser(m?.lid);
    const pn = jidUser(m?.pn);
    if (lid && pn && /^\d+$/.test(pn)) map[lid] = pn;
  }
  return map;
}

/**
 * One store row from one Baileys message, or null when there is nothing a
 * person would call a message (a read receipt, a protocol frame, a
 * reaction, a status post).
 */
export function rowFromMessage(msg, { lidToPhone } = {}) {
  const key = msg?.key;
  if (!key?.id || !key?.remoteJid) return null;
  if (key.remoteJid === 'status@broadcast' || key.remoteJid.endsWith('@broadcast')) return null;
  const content = getMessageContent(msg);
  if (!content || typeof content !== 'object') return null;

  let text = '';
  let mediaType = '';
  let media = null;
  if (typeof content.conversation === 'string') {
    text = content.conversation;
  } else if (content.extendedTextMessage?.text) {
    text = content.extendedTextMessage.text;
  }
  for (const [field, type] of MEDIA) {
    const item = content[field];
    if (!item || typeof item !== 'object') continue;
    mediaType = type;
    media = item;
    if (!text && typeof item.caption === 'string') text = item.caption;
    break;
  }
  if (!text && !mediaType) return null;

  const isGroup = key.remoteJid.endsWith('@g.us');
  const chatJid = phoneJid(key.remoteJid, key.remoteJidAlt, lidToPhone);
  const senderJid = isGroup
    ? phoneJid(key.participant, key.participantAlt, lidToPhone)
    : chatJid;

  return {
    id: String(key.id),
    chat_jid: chatJid,
    sender: jidUser(senderJid),
    content: text,
    timestamp: historyTimestamp(msg.messageTimestamp),
    is_from_me: key.fromMe ? 1 : 0,
    media_type: mediaType,
    filename: media ? String(media.fileName || '') : '',
    url: media ? String(media.url || '') : '',
    media_key: media ? bytes(media.mediaKey) : null,
    file_sha256: media ? bytes(media.fileSha256) : null,
    file_enc_sha256: media ? bytes(media.fileEncSha256) : null,
    file_length: media ? wholeNumber(media.fileLength) : null,
    direct_path: media ? String(media.directPath || '') : '',
    mimetype: media ? String(media.mimetype || '') : '',
    // Not a column: a name for the chat when nothing better is known. The
    // other person's own display name, only for a one-to-one chat and only
    // when it was they who wrote — pushName on a fromMe message is ours.
    push_name: (!isGroup && !key.fromMe && typeof msg.pushName === 'string') ? msg.pushName.trim() : '',
    is_group: isGroup,
  };
}

/** Chat rows from a history sync's `chats` (proto Conversation objects). */
export function chatRowsFromHistory(chats, { lidToPhone } = {}) {
  const rows = [];
  for (const c of chats || []) {
    const id = c?.id;
    if (!id || id === 'status@broadcast') continue;
    // The phone-number jid Baileys carries beside a LID chat, when it does.
    const jid = phoneJid(id, c.pnJid || '', lidToPhone);
    const name = String(c.name || c.displayName || '').trim();
    const last = historyTimestamp(c.conversationTimestamp ?? c.lastMessageRecvTimestamp ?? c.lastMsgTimestamp);
    rows.push({ jid, name, last_message_time: last });
  }
  return rows;
}

/** Contact rows: the name YOU saved wins, then the name they chose. */
export function contactRows(contacts, { lidToPhone } = {}) {
  const rows = [];
  for (const c of contacts || []) {
    const id = c?.id;
    if (!id) continue;
    const jid = phoneJid(id, c.phoneNumber, lidToPhone);
    const name = String(c.name || c.verifiedName || c.notify || '').trim();
    if (!name) continue;
    rows.push({ jid, name });
  }
  return rows;
}

/**
 * Open (creating if needed) the store at `dbPath`. Every method swallows
 * nothing: a store that cannot be written throws, and the caller decides
 * whether the bridge should carry on without one.
 */
export function openHistoryStore(dbPath) {
  const file = path.resolve(String(dbPath));
  mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const db = new DatabaseSync(file);
  // A reader (sqlite3 CLI, mode=ro) and this writer share the file. The
  // rollback journal, not WAL: a read-only opener of a WAL database needs
  // the -shm file to exist or be creatable, and a store left by a bridge
  // that has exited must still open read-only from anywhere.
  db.exec('PRAGMA busy_timeout = 5000;');
  db.exec(SCHEMA);

  const insertMessage = db.prepare(`
    INSERT OR IGNORE INTO messages
      (id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename,
       url, media_key, file_sha256, file_enc_sha256, file_length, direct_path, mimetype)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
  // A name is only ever replaced by a non-empty one; the last message time
  // only ever moves forward. Text comparison is correct for the fixed
  // format above.
  const upsertChat = db.prepare(`
    INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)
    ON CONFLICT(jid) DO UPDATE SET
      name = CASE WHEN excluded.name IS NOT NULL AND excluded.name != '' THEN excluded.name ELSE chats.name END,
      last_message_time = CASE
        WHEN excluded.last_message_time IS NULL THEN chats.last_message_time
        WHEN chats.last_message_time IS NULL OR excluded.last_message_time > chats.last_message_time THEN excluded.last_message_time
        ELSE chats.last_message_time END`);
  const upsertContact = db.prepare(`
    INSERT INTO contacts (jid, name) VALUES (?, ?)
    ON CONFLICT(jid) DO UPDATE SET name = excluded.name`);
  const nameChatFromContact = db.prepare(`UPDATE chats SET name = ? WHERE jid = ?`);
  const contactName = db.prepare(`SELECT name FROM contacts WHERE jid = ?`);
  const chatName = db.prepare(`SELECT name FROM chats WHERE jid = ?`);
  const mediaRow = db.prepare(`
    SELECT media_type, url, direct_path, media_key, mimetype, filename, file_length
    FROM messages WHERE id = ? AND chat_jid = ?`);
  const countRow = db.prepare(`SELECT (SELECT count(*) FROM messages) AS messages, (SELECT count(*) FROM chats) AS chats`);

  function inTransaction(fn) {
    db.exec('BEGIN');
    try {
      const out = fn();
      db.exec('COMMIT');
      return out;
    } catch (err) {
      try { db.exec('ROLLBACK'); } catch {}
      throw err;
    }
  }

  return {
    path: file,

    /** Record a batch of Baileys messages. Returns how many were new. */
    recordMessages(msgs, opts = {}) {
      const rows = [];
      for (const m of msgs || []) {
        const r = rowFromMessage(m, opts);
        if (r) rows.push(r);
      }
      if (!rows.length) return 0;
      return inTransaction(() => {
        let added = 0;
        for (const r of rows) {
          // The chat row first (the FK), named as well as it can be: a saved
          // contact, else whatever is already there, else their push name.
          let name = '';
          const saved = contactName.get(r.chat_jid);
          if (saved?.name) name = saved.name;
          else if (r.push_name && !(chatName.get(r.chat_jid)?.name)) name = r.push_name;
          upsertChat.run(r.chat_jid, name, r.timestamp);
          const res = insertMessage.run(
            r.id, r.chat_jid, r.sender, r.content, r.timestamp, r.is_from_me, r.media_type,
            r.filename, r.url, r.media_key, r.file_sha256, r.file_enc_sha256, r.file_length,
            r.direct_path, r.mimetype,
          );
          added += Number(res.changes) || 0;
        }
        return added;
      });
    },

    /** Chats from a history sync or a chats.upsert/update. */
    recordChats(chats, opts = {}) {
      const rows = chatRowsFromHistory(chats, opts);
      if (!rows.length) return 0;
      return inTransaction(() => {
        for (const r of rows) upsertChat.run(r.jid, r.name, r.last_message_time);
        return rows.length;
      });
    },

    /** Contacts: remembered, and every chat they name is renamed now. */
    recordContacts(contacts, opts = {}) {
      const rows = contactRows(contacts, opts);
      if (!rows.length) return 0;
      return inTransaction(() => {
        for (const r of rows) {
          upsertContact.run(r.jid, r.name);
          nameChatFromContact.run(r.name, r.jid);
        }
        return rows.length;
      });
    },

    /** A group's subject, from groups.upsert / groups.update. */
    nameChat(jid, name) {
      const n = String(name || '').trim();
      if (!jid || !n) return;
      upsertChat.run(String(jid), n, null);
    },

    /** What is needed to download one message's media, or null. */
    mediaRef(messageId, chatJid) {
      const row = mediaRow.get(String(messageId || ''), String(chatJid || ''));
      if (!row || !row.media_type) return null;
      return {
        type: row.media_type,
        url: row.url || '',
        directPath: row.direct_path || '',
        mediaKey: row.media_key || null,
        mimetype: row.mimetype || '',
        fileName: row.filename || '',
        fileLength: row.file_length ?? null,
      };
    },

    counts() {
      const r = countRow.get();
      return { messages: Number(r?.messages || 0), chats: Number(r?.chats || 0) };
    },

    close() {
      try { db.close(); } catch {}
    },
  };
}
