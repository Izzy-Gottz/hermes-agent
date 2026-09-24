// Runs the REAL extensions/chrome/background.js under node against a fake chrome.* and plays
// the native host: sends it ops, records what it asked Chrome to do, prints one JSON report.
// Driven by tests/tools/test_chrome_extension_js.py.
"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const SRC = process.argv[2] || path.join(__dirname, "..", "..", "extensions", "chrome", "background.js");
const NONE = -1;

// Tab 1 is the person's own tab (no group); everything Moe makes gets ids from 100.
const tabs = new Map([[1, { id: 1, windowId: 10, groupId: NONE, url: "https://mail.example/inbox", status: "complete" }]]);
const groups = new Map();
const windows = new Map([[10, { id: 10, focused: true }]]);
let nextTab = 100, nextGroup = 500, nextWindow = 20;
const calls = { attach: [], detach: [], send: [], tabsCreate: [], windowsCreate: [], windowsUpdate: [], tabsUpdate: [] };
const listeners = {};
const on = (name) => ({ addListener: (fn) => { (listeners[name] = listeners[name] || []).push(fn); } });
let hookOnSend = null;

const store = {};
const chrome = {
  runtime: {
    id: "ljblhmlmgflmbffbfelamefleikodmjh",
    getManifest: () => ({ version: "test" }),
    connectNative: () => port,
    onStartup: on("onStartup"), onInstalled: on("onInstalled"),
  },
  extension: { isAllowedFileSchemeAccess: (cb) => cb(false) },
  action: { onClicked: on("action") },
  storage: { session: {
    get: async (k) => (k in store ? { [k]: store[k] } : {}),
    set: async (o) => { Object.assign(store, o); },
  } },
  tabGroups: {
    TAB_GROUP_ID_NONE: NONE,
    get: async (id) => { if (!groups.has(id)) throw new Error("no group"); return groups.get(id); },
    update: async (id, props) => Object.assign(groups.get(id), props),
    onRemoved: on("groupRemoved"),
  },
  tabs: {
    get: async (id) => { if (!tabs.has(id)) throw new Error(`No tab with id: ${id}`); return { ...tabs.get(id) }; },
    create: async ({ windowId, url, active }) => {
      calls.tabsCreate.push({ windowId, url, active });
      const t = { id: nextTab++, windowId, groupId: NONE, url, status: "complete" };
      tabs.set(t.id, t);
      return { ...t };
    },
    group: async ({ tabIds, groupId, createProperties }) => {
      let gid = groupId;
      if (gid == null) { gid = nextGroup++; groups.set(gid, { id: gid, windowId: createProperties.windowId }); }
      for (const id of tabIds) tabs.get(id).groupId = gid;
      return gid;
    },
    update: async (id, props) => { calls.tabsUpdate.push({ id, props }); return tabs.get(id); },
    remove: async (id) => { tabs.delete(id); },
    onUpdated: on("tabsUpdated"), onRemoved: on("tabsRemoved"),
  },
  windows: {
    get: async (id) => { if (!windows.has(id)) throw new Error("no window"); return windows.get(id); },
    getLastFocused: async () => [...windows.values()].find((w) => w.focused) || windows.get(10),
    create: async ({ url, focused }) => {
      calls.windowsCreate.push({ url, focused });
      const w = { id: nextWindow++, focused: false };
      windows.set(w.id, w);
      const t = { id: nextTab++, windowId: w.id, groupId: NONE, url, status: "complete" };
      tabs.set(t.id, t);
      return { ...w, tabs: [{ ...t }] };
    },
    update: async (id, props) => { calls.windowsUpdate.push({ id, props }); return windows.get(id); },
  },
  debugger: {
    getTargets: async () => [...tabs.values()].map((t) => ({ id: `T${t.id}`, tabId: t.id, type: "page", url: t.url, title: "" })),
    attach: async ({ tabId }) => { calls.attach.push(tabId); },
    detach: async ({ tabId }) => { calls.detach.push(tabId); },
    sendCommand: async ({ tabId }, method, params) => {
      if (method === "Page.getFrameTree") {
        const t = tabs.get(tabId);
        return { frameTree: { frame: { id: `T${tabId}`, url: t.shown || t.url, unreachableUrl: t.unreachable } } };
      }
      calls.send.push({ tabId, method, params });
      if (method === "Page.navigate") tabs.get(tabId).url = params.url;
      if (hookOnSend) hookOnSend(tabId, method, params);
      return method === "Runtime.evaluate" ? { result: { value: "secret-from-" + tabs.get(tabId).url } } : {};
    },
    onEvent: on("dbgEvent"), onDetach: on("dbgDetach"),
  },
};

const replies = new Map();
const port = {
  onMessage: { addListener: (fn) => { port._deliver = fn; } },
  onDisconnect: { addListener: () => {} },
  postMessage: (m) => { if ("id" in m) replies.set(m.id, m); },
};

const ctx = vm.createContext({ chrome, navigator: { userAgent: "Chrome/153 test" }, URL, setTimeout, console });
vm.runInContext(fs.readFileSync(SRC, "utf8"), ctx, { filename: SRC });

let n = 0;
async function op(name, args) {
  const id = ++n;
  port._deliver({ id, op: name, args });
  for (let i = 0; i < 400 && !replies.has(id); i++) await new Promise((r) => setTimeout(r, 1));
  const r = replies.get(id) || { timeout: true };
  return r.error ? { error: r.error.code, message: r.error.message } : { result: r.result };
}
const sentTo = (tabId) => calls.send.filter((c) => c.tabId === tabId);

(async () => {
  const report = {};
  report.hello = await op("hello", {});

  // A Moe tab: its own window, behind the person's (the person's window gets focus handed back).
  report.create = await op("create", { url: "https://example.com/" });
  const moe = report.create.result && report.create.result.tabId;
  report.createWindows = calls.windowsCreate.slice();
  report.refocus = calls.windowsUpdate.slice();
  report.moeNavigate = sentTo(moe).map((c) => [c.method, c.params && c.params.url]);

  // URL policy on createTab.
  report.createRefused = {};
  for (const url of ["file:///etc/hosts", "chrome://settings", "chrome-extension://ljblhmlmgflmbffbfelamefleikodmjh/manifest.json",
                     "javascript:alert(1)", "view-source:https://example.com/", "data:text/html,hi"]) {
    const before = tabs.size;
    const r = await op("create", { url });
    report.createRefused[url] = { error: r.error || null, tabsMade: tabs.size - before };
  }

  // The person's tab (1, no group): every op refused, chrome.debugger never touched for it.
  report.foreign = {
    cdp: await op("cdp", { tabId: 1, method: "Runtime.evaluate", params: { expression: "document.cookie" } }),
    attach: await op("attach", { tabId: 1 }),
    close: await op("close", { tabId: 1 }),
  };
  report.foreignDebuggerCalls = { attach: calls.attach.filter((t) => t === 1).length, send: sentTo(1).length };
  report.foreignStillOpen = tabs.has(1);

  // Moe's tab: allowed read works; navigation off-policy refused.
  report.moeRead = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "1" } });
  report.moeNavFile = await op("cdp", { tabId: moe, method: "Page.navigate", params: { url: "file:///etc/hosts" } });

  // A tab in Moe's group that is somewhere forbidden (e.g. dragged in from chrome://): commands
  // refused, and the tab is LEFT where it is; Moe may still navigate it to an allowed URL.
  tabs.get(moe).url = "file:///etc/hosts";
  const mark = calls.send.length;
  report.moeOnFile = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "document.body.innerText" } });
  report.moeOnFileSends = calls.send.slice(mark).map((c) => [c.method, c.params && c.params.url]);
  report.moeUrlAfter = tabs.get(moe).url;
  report.moeNavAway = await op("cdp", { tabId: moe, method: "Page.navigate", params: { url: "https://example.com/" } });
  report.moeReadAfterNavAway = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "1" } });

  // An error page: getTargets still names the http URL, the frame shows chrome-error with unreachableUrl.
  tabs.get(moe).unreachable = "https://nx.invalid/";
  report.moeErrorPage = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "document.body.innerText" } });
  delete tabs.get(moe).unreachable;
  tabs.get(moe).shown = "chrome-error://chromewebdata/";
  report.moeChromeErrorShown = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "1" } });
  delete tabs.get(moe).shown;

  // A Moe tab that navigates to a forbidden URL DURING a command: the result is withheld.
  hookOnSend = (tabId, method) => { if (method === "Runtime.evaluate") tabs.get(tabId).url = "chrome://settings"; };
  report.moeRace = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "x" } });
  hookOnSend = null;

  // Nothing the host sends can focus a window or a tab.
  const updatesBefore = calls.windowsUpdate.length + calls.tabsUpdate.length;
  report.activate = await op("activate", { tabId: moe });
  report.focusCallsAfterActivate = calls.windowsUpdate.length + calls.tabsUpdate.length - updatesBefore;

  // Dragged out of the group: the tab is the person's again.
  tabs.get(moe).url = "https://example.com/";
  tabs.get(moe).groupId = NONE;
  for (const fn of listeners.tabsUpdated || []) await fn(moe, { groupId: NONE });
  report.afterDragOut = await op("cdp", { tabId: moe, method: "Runtime.evaluate", params: { expression: "1" } });

  // targets lists only Moe's group.
  const second = (await op("create", { url: "https://example.org/" })).result.tabId;
  report.targets = (await op("targets", {})).result.map((t) => t.tabId);
  report.secondSameWindow = tabs.get(second).windowId === tabs.get(moe).windowId;

  process.stdout.write(JSON.stringify(report));
})().catch((e) => { process.stdout.write(JSON.stringify({ crashed: String(e && e.stack || e) })); });
