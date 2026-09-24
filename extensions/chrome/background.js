// Memoe — Moe in your Chrome. MV3 service worker.
//
// Moe drives the owner's REAL Chrome, but only inside a tab group of its own
// ("Moe"). The rule the whole extension enforces, in one place:
//
//     A tab is Moe's if and only if it sits in a group this extension created.
//
// Moe opens its tabs in that group, in a window of Moe's own behind the person's,
// so the person's window keeps focus (a never-shown background tab in their own
// window stalls: measured on Chrome 153). Nothing here ever focuses a window. The person can drag any tab INTO the group to share it with
// Moe ("look at this"), and drag a tab OUT to take it back — the debugger lets
// go the moment it leaves. Every command naming a tab outside the group is
// refused here, before chrome.debugger is ever called.
//
// Input is chrome.debugger (CDP Input.dispatchMouseEvent / dispatchKeyEvent),
// so clicks and keys arrive with isTrusted=true — content-script events are
// isTrusted=false, which is exactly what PerimeterX-class scripts check.
// The price is Chrome's "Memoe started debugging this browser" bar while a tab
// is attached; its Cancel button detaches everything (the owner's off switch).
//
// Transport: native messaging to the host `app.memoe.chrome_bridge`. Only an
// extension whose id is in the host manifest's allowed_origins can start it,
// and no web page can reach either end — there is no open port on this side.

const HOST = "app.memoe.chrome_bridge";
const GROUP_TITLE = "Moe";
const GROUP_COLOR = "purple";
const PROTOCOL = 1;

// The lane's one URL policy, the same as the relay's url_allowed(): http(s) and about:blank.
// Checked on every tab Moe opens, on every navigation it sends, and on the tab's current URL
// before and after every command — so a Moe tab that is somehow elsewhere is never read.
function urlAllowed(url) {
  if (typeof url !== "string") return false;
  if (url === "about:blank" || url.startsWith("about:blank#") || url.startsWith("about:blank?")) return true;
  let u;
  try { u = new URL(url); } catch { return false; }
  return (u.protocol === "http:" || u.protocol === "https:") && !!u.hostname;
}

function refuseUrl(url) {
  const err = new Error(`${String(url).slice(0, 80)} is refused in the owner's Chrome: Moe's tabs open only http, https and about:blank`);
  err.code = "url_refused";
  return err;
}

let port = null;
let retryMs = 1000;
const attached = new Set(); // tabIds this extension has chrome.debugger attached to

// ---- ownership -------------------------------------------------------------

async function ownedGroupIds() {
  const { moeGroups = [] } = await chrome.storage.session.get("moeGroups");
  return new Set(moeGroups);
}

async function rememberGroup(groupId) {
  const groups = await ownedGroupIds();
  groups.add(groupId);
  await chrome.storage.session.set({ moeGroups: [...groups] });
}

async function forgetGroup(groupId) {
  const groups = await ownedGroupIds();
  if (groups.delete(groupId)) await chrome.storage.session.set({ moeGroups: [...groups] });
}

async function isOwned(tabId) {
  if (typeof tabId !== "number") return false;
  let tab;
  try { tab = await chrome.tabs.get(tabId); } catch { return false; }
  if (tab.groupId === chrome.tabGroups.TAB_GROUP_ID_NONE) return false;
  return (await ownedGroupIds()).has(tab.groupId);
}

async function requireOwned(tabId) {
  if (!(await isOwned(tabId))) {
    const err = new Error(`tab ${tabId} is not in Moe's tab group; Moe never touches the person's own tabs`);
    err.code = "not_owned";
    throw err;
  }
}

async function groupInWindow(windowId) {
  for (const gid of await ownedGroupIds()) {
    try {
      const g = await chrome.tabGroups.get(gid);
      if (g.windowId === windowId) return gid;
    } catch { await forgetGroup(gid); }
  }
  return null;
}

async function currentUrl(tabId) {
  const t = (await chrome.debugger.getTargets()).find((x) => x.tabId === tabId);
  return t ? t.url : null;
}

async function committedUrl(tabId) {
  // What the tab's main frame actually shows. getTargets() can still name the URL that was asked
  // for while the frame shows chrome-error://chromewebdata/ (security re-review, 2026-09-23); an
  // error page carries the failed URL in unreachableUrl.
  try {
    const { frameTree } = await chrome.debugger.sendCommand({ tabId }, "Page.getFrameTree", {});
    const f = (frameTree && frameTree.frame) || {};
    return f.unreachableUrl ? `chrome-error://${f.unreachableUrl}` : f.url;
  } catch {
    return null;
  }
}

async function requireAllowedUrl(tabId) {
  // Somewhere Moe may not be — refuse the command, and leave the tab where it is: a tab the
  // person dragged into the group from chrome:// (or anywhere else) is theirs to keep, not Moe's
  // to discard by navigating it away. Moe can still send it to an allowed URL (Page.navigate).
  const listed = await currentUrl(tabId);
  if (!urlAllowed(listed)) throw refuseUrl(listed);
  const shown = await committedUrl(tabId);
  if (!urlAllowed(shown)) throw refuseUrl(shown);
}

async function ownedTargets() {
  const targets = await chrome.debugger.getTargets();
  const out = [];
  for (const t of targets) {
    if (t.type !== "page" || typeof t.tabId !== "number") continue;
    if (!(await isOwned(t.tabId))) continue;
    if (!urlAllowed(t.url)) continue;
    out.push({ targetId: t.id, tabId: t.tabId, url: t.url || "", title: t.title || "", attached: attached.has(t.tabId) });
  }
  return out;
}

async function targetIdFor(tabId) {
  for (let i = 0; i < 20; i++) {
    const t = (await chrome.debugger.getTargets()).find((x) => x.tabId === tabId);
    if (t) return t.id;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no debugger target for tab ${tabId}`);
}

// ---- operations the host may ask for ----------------------------------------

async function moeWindow() {
  const { moeWindow: id } = await chrome.storage.session.get("moeWindow");
  if (typeof id !== "number") return null;
  try { return await chrome.windows.get(id); } catch { return null; }
}

async function createTab({ url = "about:blank" }) {
  // Owner's decision (2026-09-23): Moe's group lives in a window of its own, behind the
  // person's. Its tab is that window's active tab, so Chrome renders it (loads, input and
  // screenshots work) while the person's window keeps focus.
  if (!urlAllowed(url)) throw refuseUrl(url);
  let tab;
  const win = await moeWindow();
  if (win) {
    tab = await chrome.tabs.create({ windowId: win.id, url: "about:blank", active: true });
  } else {
    // macOS raises a new window even with focused:false (measured: it came up first in the
    // window stack). If the person was in Chrome, hand their window straight back; if they
    // were in another app, Chrome was not focused and nothing is touched.
    let before = null;
    try { before = await chrome.windows.getLastFocused({ windowTypes: ["normal"] }); } catch { /* none */ }
    const w = await chrome.windows.create({ url: "about:blank", focused: false, width: 1280, height: 900 });
    await chrome.storage.session.set({ moeWindow: w.id });
    if (before && before.focused) {
      try { await chrome.windows.update(before.id, { focused: true }); } catch { /* closed meanwhile */ }
    }
    tab = w.tabs[0];
  }
  let gid = await groupInWindow(tab.windowId);
  if (gid == null) {
    gid = await chrome.tabs.group({ tabIds: [tab.id], createProperties: { windowId: tab.windowId } });
    await chrome.tabGroups.update(gid, { title: GROUP_TITLE, color: GROUP_COLOR });
    await rememberGroup(gid);
  } else {
    await chrome.tabs.group({ tabIds: [tab.id], groupId: gid });
  }
  // A freshly created tab can sit "loading" with no committed document (Runtime.evaluate then
  // never answers) until something navigates it; a debugger navigation commits it.
  await attach({ tabId: tab.id });
  await chrome.debugger.sendCommand({ tabId: tab.id }, "Page.navigate", { url });
  await settled(tab.id);
  return { tabId: tab.id, targetId: await targetIdFor(tab.id) };
}

async function settled(tabId) {
  // A background tab's initial about:blank can sit with no committed document, and
  // Runtime.evaluate then waits forever for a context. Wait for "complete" first.
  for (let i = 0; i < 100; i++) {
    const t = await chrome.tabs.get(tabId);
    if (i === 0 || t.status === "complete") log("tab", tabId, "status", t.status, "discarded", t.discarded, "frozen", t.frozen);
    if (t.status === "complete") return;
    await new Promise((r) => setTimeout(r, 50));
  }
}

async function attach({ tabId }) {
  await requireOwned(tabId);
  if (!attached.has(tabId)) {
    await chrome.debugger.attach({ tabId }, "1.3");
    attached.add(tabId);
  }
  return {};
}

async function detach({ tabId }) {
  if (attached.delete(tabId)) {
    try { await chrome.debugger.detach({ tabId }); } catch { /* already gone */ }
  }
  return {};
}

async function cdp({ tabId, method, params }) {
  await requireOwned(tabId);
  if (method === "Page.navigate" && !urlAllowed((params || {}).url)) throw refuseUrl((params || {}).url);
  if (!attached.has(tabId)) await attach({ tabId });
  // Navigating away from a forbidden page reads nothing from it, so only navigation skips the
  // before-check (its URL was checked above); every other command needs an allowed page.
  if (method !== "Page.navigate") await requireAllowedUrl(tabId);
  const result = (await chrome.debugger.sendCommand({ tabId }, method, params || {})) || {};
  // A result read from a tab that ended up somewhere Moe may not be is never handed back.
  if (method !== "Page.navigate") await requireAllowedUrl(tabId);
  return result;
}

async function closeTab({ tabId }) {
  await requireOwned(tabId);
  await detach({ tabId });
  await chrome.tabs.remove(tabId);
  return {};
}

async function hello() {
  return {
    protocol: PROTOCOL,
    extensionId: chrome.runtime.id,
    version: chrome.runtime.getManifest().version,
    userAgent: navigator.userAgent,
    // "Allow access to file URLs" is a per-extension switch in chrome://extensions that a
    // manifest cannot set. Measured: TRUE for an unpacked load (--load-extension, Chrome for
    // Testing 153); a Web Store install starts with it off. Reported so the app can tell the
    // person to switch it off — the URL policy above refuses file: either way.
    fileAccess: await new Promise((r) => chrome.extension.isAllowedFileSchemeAccess(r)),
  };
}

// No "activate": nothing the host can send focuses a window or a tab.
const OPS = { hello, targets: ownedTargets, create: createTab, attach, detach, cdp, close: closeTab };

// ---- native messaging --------------------------------------------------------

function log(...parts) {
  post({ event: "log", text: parts.join(" ") });
}

function post(msg) {
  try { port && port.postMessage(msg); } catch { /* port died; onDisconnect reconnects */ }
}

async function onHostMessage(msg) {
  const { id, op } = msg || {};
  log("op", id, op, JSON.stringify(msg.args || {}).slice(0, 160));
  const fn = OPS[op];
  if (!fn) return post({ id, error: { code: "unknown_op", message: `unknown op ${op}` } });
  try {
    const result = await fn(msg.args || {});
    log("done", id, op);
    post({ id, result });
  } catch (e) {
    log("fail", id, op, String(e && e.message || e));
    post({ id, error: { code: e.code || "failed", message: String(e && e.message || e) } });
  }
}

function connect() {
  if (port) return;
  try {
    port = chrome.runtime.connectNative(HOST);
  } catch (e) {
    port = null;
    setTimeout(connect, (retryMs = Math.min(retryMs * 2, 60000)));
    return;
  }
  port.onMessage.addListener(onHostMessage);
  port.onDisconnect.addListener(() => {
    port = null;
    for (const tabId of [...attached]) detach({ tabId });
    setTimeout(connect, (retryMs = Math.min(retryMs * 2, 60000)));
  });
  retryMs = 1000;
}

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (typeof source.tabId === "number" && attached.has(source.tabId)) {
    post({ event: "cdp", tabId: source.tabId, method, params });
  }
});

chrome.debugger.onDetach.addListener((source, reason) => {
  // reason "canceled_by_user" = the person pressed Cancel on the debugging bar.
  if (attached.delete(source.tabId)) post({ event: "detached", tabId: source.tabId, reason });
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo) => {
  if (!("groupId" in changeInfo) || !attached.has(tabId)) return;
  if (!(await isOwned(tabId))) {
    // Dragged out of the Moe group: it is the person's again, immediately.
    await detach({ tabId });
    post({ event: "released", tabId, reason: "left_group" });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (attached.delete(tabId)) post({ event: "released", tabId, reason: "closed" });
});

chrome.tabGroups.onRemoved.addListener((group) => { forgetGroup(group.id); });
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
chrome.action.onClicked.addListener(connect);
connect();
