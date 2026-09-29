"""Moe ticket #17: accounts Hermes can make and keep without anyone seeing a password.

* the login is bound to the page the model is on, never the first tab with a password box;
* a saved login for the exact site is used before anything is created;
* a signup password is generated inside the tool, stored pending, put into every new/confirm
  field and never returned, logged, put in argv or the environment;
* pending logins are confirmed, regenerated (only while pending) and pruned when stale;
* the host's own prompt program (HERMES_SECRET_PROMPT_CMD) and its one-JSON-in, one-JSON-out
  contract, used only when the person is at the Mac on a live turn;
* on a turn nobody is at: PERSON_NEEDED, nothing created.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore

FAZIER = "https://fazier.com"
IH = "https://www.indiehackers.com"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


@pytest.fixture(autouse=True)
def only_local_backend_and_clean_prompts(store, tmp_path, monkeypatch):
    """The local vault only (no real 1Password/Bitwarden on the test Mac), no prompt left installed,
    no host prompt program unless a test sets one, and a private HERMES_HOME."""
    from agent.vault_backends import unlock as unlock_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_SECRET_PROMPT_CMD", raising=False)
    for setter in (unlock_mod.set_unlock_prompt_callback, unlock_mod.set_save_login_prompt_callback,
                   unlock_mod.set_code_prompt_callback):
        setter(None)
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch("agent.vault_backends.base.is_enabled", return_value=False):
        yield
    for setter in (unlock_mod.set_unlock_prompt_callback, unlock_mod.set_save_login_prompt_callback,
                   unlock_mod.set_code_prompt_callback):
        setter(None)


def _live(live=True):
    return patch("tools.browser_chrome_extension.unattended_turn", return_value=None if live else "a scheduled job")


SIGNUP_CONTROLS = [
    {"index": 0, "type": "email", "name": "email", "label": "Email", "autocomplete": "email", "formIndex": 0},
    {"index": 1, "type": "password", "name": "password", "label": "Password", "autocomplete": "new-password",
     "formIndex": 0, "minLength": 12, "maxLength": 16,
     "passwordRules": "required: upper; required: digit; allowed: lower, [-_];"},
    {"index": 2, "type": "password", "name": "confirm_password", "label": "Confirm password", "autocomplete": "",
     "formIndex": 0},
]
LOGIN_CONTROLS = [
    {"index": 0, "type": "email", "name": "email", "label": "Email", "autocomplete": "username", "formIndex": 0},
    {"index": 1, "type": "password", "name": "password", "label": "Password", "autocomplete": "current-password",
     "formIndex": 0},
]


class Page:
    """The page the vault tool sees: controls for inspection, a URL, and every secret-bearing script."""

    def __init__(self, url=FAZIER + "/signup", controls=SIGNUP_CONTROLS, fill_error=None, filled=None):
        self.url, self.controls, self.fill_error, self.filled = url, controls, fill_error, filled
        self.plain_exprs, self.secret_exprs = [], []

    def eval(self, task_id, expr):
        self.plain_exprs.append(expr)
        if "location.href" in expr:
            return {"success": True, "result": self.url}
        if 'querySelectorAll("input, select")' in expr:
            return {"success": True, "result": json.dumps(self.controls)}
        return {"success": True, "result": None}

    def eval_secret(self, task_id, expr):
        self.secret_exprs.append(expr)
        if self.fill_error is not None:
            raise self.fill_error(expr)
        fills = self.fills()
        return {"success": True, "result": json.dumps({"filled": self.filled if self.filled is not None else len(fills)})}

    def fills(self, i=-1):
        m = re.search(r"const fills = (\[.*?\]);\n", self.secret_exprs[i])
        return json.loads(m.group(1))

    def patches(self):
        from tools import browser_vault_tool as bvt
        return (patch.object(bvt, "_ensure_supervisor", return_value=None),
                patch.object(bvt, "_eval_js", side_effect=self.eval),
                patch.object(bvt, "_eval_js_secret", side_effect=self.eval_secret))


def _run(page, fn, *args, **kw):
    a, b, c = page.patches()
    with a, b, c:
        return fn(*args, **kw)


# ---------------------------------------------------------------------------
# 1. origin binding: the two-tab mix-up
# ---------------------------------------------------------------------------

class TabsSupervisor:
    """A CDPSupervisor whose CDP is two open tabs, both with a password field (the ticket's browser:
    a stale indiehackers.com tab first, the fazier.com page the model was on second)."""

    def __new__(cls, tabs):
        from tools.browser_supervisor import CDPSupervisor

        class _Sup(CDPSupervisor):
            async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
                if method == "Target.getTargets":
                    return {"result": {"targetInfos": [{"targetId": t, "type": "page", "url": u} for t, u, _ in self.tabs]}}
                if method == "Target.attachToTarget":
                    return {"result": {"sessionId": "S-" + params["targetId"]}}
                if method == "Runtime.evaluate":
                    has = {t: form for t, _u, form in self.tabs}[session_id[2:]]
                    return {"result": {"result": {"value": has}}}
                return {"result": {}}

            async def _enable_page_domains(self, session_id, *, timeout):
                return None

            async def _install_dialog_bridge(self, session_id):
                return None

            def current_url(self):
                sid = self._page_session_id
                return {t: u for t, u, _ in self.tabs}[sid[2:]] if sid else "about:blank"

        sup = _Sup("t", "ws://127.0.0.1:9/devtools/browser/fake")
        sup.tabs = tabs
        sup._loop = asyncio.new_event_loop()
        threading.Thread(target=sup._loop.run_forever, daemon=True).start()
        return sup


TWO_TABS = [("T-ih", IH + "/sign-in", True), ("T-fz", FAZIER + "/login", True)]


def _record_harness_tab(target_id, url, at=None):
    from tools import browser_exec_health as health
    path = Path(health.current_tab_path("t"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"targetId": target_id, "url": url, "at": time.time() if at is None else at}))


class TestOriginBinding:
    def test_first_tab_with_a_password_field_no_longer_wins(self):
        sup = TabsSupervisor(TWO_TABS)
        from tools.browser_vault_tool import _TAB_PROBES
        out = sup.focus_page("", accept=_TAB_PROBES["login"])
        assert out["ok"] is False and out["ambiguous"] == [IH, FAZIER]
        assert sup._page_session_id is None  # nothing was re-pointed
        assert sup.focus_page("", accept=_TAB_PROBES["login"], target_id="T-fz")["url"] == FAZIER + "/login"
        assert sup.current_url() == FAZIER + "/login"
        assert sup.focus_page(IH, accept=_TAB_PROBES["login"])["ok"] and sup.current_url().startswith(IH)

    def _save(self, sup, **kw):
        from tools import browser_vault_tool as bvt
        from agent.vault_backends import unlock as unlock_mod

        asked = []
        unlock_mod.set_save_login_prompt_callback(
            lambda origin, host: asked.append(origin) or {"identifier": "tek@fazier.test", "password": "pw-Fazier-1"})
        href = lambda t, e: {"success": True, "result": sup.current_url()}
        with patch.object(bvt, "_ensure_supervisor", return_value=sup), \
             patch.object(bvt, "_eval_js", side_effect=href), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=True), \
             patch.object(bvt, "browser_vault_fill", lambda handle, task_id=None: json.dumps({"success": True})):
            out = json.loads(bvt.browser_vault_save_login(label="Fazier", task_id="t", **kw))
        return out, asked

    def test_save_login_binds_the_tab_browser_exec_is_on(self, store):
        """The ticket: save_login(label='Fazier') on fazier.com bound indiehackers.com."""
        _record_harness_tab("T-fz", FAZIER + "/login")
        out, asked = self._save(TabsSupervisor(TWO_TABS))
        assert out["success"] and out["origin"] == FAZIER and asked == [FAZIER]
        assert [m.origin for m in store.list_items()] == [FAZIER]

    def test_explicit_url_binds_that_site(self, store):
        out, asked = self._save(TabsSupervisor(TWO_TABS), url=FAZIER + "/login")
        assert out["origin"] == FAZIER and asked == [FAZIER]

    def test_unknowable_tab_is_refused_and_nothing_is_saved(self, store):
        out, asked = self._save(TabsSupervisor(TWO_TABS))
        assert out["error_type"] == "ambiguous_page" and set(out["open_sites"]) == {IH, FAZIER}
        assert asked == [] and store.list_items() == []

    def test_url_on_a_site_with_no_open_tab_is_refused(self, store):
        sup = TabsSupervisor([("T-ih", IH + "/sign-in", True)])
        sup.focus_page("", target_id="T-ih")
        out, asked = self._save(sup, url=FAZIER + "/login")
        assert out["error_type"] == "origin_mismatch" and asked == [] and store.list_items() == []

    def test_a_stale_harness_record_is_ignored(self, store):
        _record_harness_tab("T-fz", FAZIER + "/login", at=time.time() - 3 * 3600)
        out, _ = self._save(TabsSupervisor(TWO_TABS))
        assert out["error_type"] == "ambiguous_page"

    def test_the_harness_records_its_tab_path_in_moes_own_browser_only(self, tmp_path):
        from tools import browser_exec_health as health
        env = {}
        health.exec_env(env, timeout_s=60, session="", own_lane=True, task_id="t")
        assert env[health.ENV_CURRENT_TAB] == health.current_tab_path("t")
        other = {}
        health.exec_env(other, timeout_s=60, session="", own_lane=False, task_id="t")
        assert health.ENV_CURRENT_TAB not in other
        assert "_record_current_tab" in health.HARNESS_PATCH_SOURCE and "current_tab" in health.HARNESS_PATCH_SOURCE


# ---------------------------------------------------------------------------
# 2. lookup first
# ---------------------------------------------------------------------------

class TestLookupFirst:
    def _existing(self, store, origin=FAZIER, ident="tek@fazier.test"):
        return store.add_item("login", "Fazier", {"identifier_type": "email", "identifier": ident,
                                                  "password": "old-Secret-9"}, origin=origin)

    def test_login_uses_the_saved_login_and_asks_nobody(self, store):
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool as bvt

        meta = self._existing(store)
        asked = []
        unlock_mod.set_save_login_prompt_callback(lambda *a: asked.append(a) or {"identifier": "x", "password": "y"})
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        with patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
            raw = _run(page, bvt.browser_vault_login, FAZIER + "/login", mode="login", task_id="t")
        out = json.loads(raw)
        assert out["outcome"] == "used_existing" and out["handle"] == meta.id and out["success"]
        assert out["identifier"] == "tek@fazier.test" and asked == []
        assert page.fills() == [{"index": 1, "token": "current-password", "value": "old-Secret-9"}]
        assert "old-Secret-9" not in raw and len(store.list_items()) == 1

    def test_save_login_also_looks_first(self, store):
        from tools import browser_vault_tool as bvt

        self._existing(store)
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        out = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER))
        assert out["outcome"] == "used_existing" and len(store.list_items()) == 1

    def test_a_login_for_another_site_is_not_used(self, store):
        from tools import browser_vault_tool as bvt

        self._existing(store, origin=IH)
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        with _live(True):
            out = json.loads(_run(page, bvt.browser_vault_login, FAZIER, mode="login", task_id="t"))
        assert out["outcome"] != "used_existing" and page.secret_exprs == []

    def test_signup_on_a_site_with_a_saved_login_creates_nothing(self, store):
        from tools import browser_vault_tool as bvt

        self._existing(store)
        page = Page()
        with _live(True):
            out = json.loads(_run(page, bvt.browser_vault_login, FAZIER, mode="signup", username="new@x.test", task_id="t"))
        assert out["outcome"] == "existing_account" and len(store.list_items()) == 1 and page.secret_exprs == []

    def test_several_saved_logins_ask_which(self, store):
        from tools import browser_vault_tool as bvt

        self._existing(store)
        self._existing(store, ident="work@fazier.test")
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        out = json.loads(_run(page, bvt.browser_vault_login, FAZIER, mode="login", task_id="t"))
        assert out["outcome"] == "choose" and len(out["logins"]) == 2 and page.secret_exprs == []
        picked = json.loads(_run(page, bvt.browser_vault_login, FAZIER, mode="login", username="work@fazier.test", task_id="t"))
        assert picked["outcome"] == "used_existing" and picked["identifier"] == "work@fazier.test"


# ---------------------------------------------------------------------------
# 3. generate
# ---------------------------------------------------------------------------

def _no_secret_anywhere(secret, *texts):
    for text in texts:
        assert secret not in str(text)
    assert all(secret not in v for v in os.environ.values())


class TestGenerate:
    def test_generated_password_obeys_the_page_fills_both_fields_and_is_never_exposed(self, store, caplog):
        from tools import browser_vault_tool as bvt

        page = Page()
        spawned = []
        real_popen = subprocess.Popen

        def spy(*a, **k):
            spawned.append(a[0] if a else k.get("args"))
            return real_popen(*a, **k)

        caplog.set_level(logging.DEBUG)
        with _live(True), patch("subprocess.Popen", side_effect=spy):
            raw = _run(page, bvt.browser_vault_save_login, label="Fazier", task_id="t", url=FAZIER + "/signup",
                       username="tek@fazier.test", generate=True)
        out = json.loads(raw)
        [meta] = store.list_items()
        password = store.resolve_secret(meta.id)["password"]

        fills = page.fills()
        assert [f["index"] for f in fills] == [1, 2] and {f["value"] for f in fills} == {password}
        assert {f["token"] for f in fills} == {"new-password"}
        assert 12 <= len(password) <= 16
        assert re.search(r"[A-Z]", password) and re.search(r"\d", password)
        assert set(password) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

        assert out == {**out, "success": True, "outcome": "created", "handle": meta.id, "origin": FAZIER,
                       "generated": True, "pending": True, "identifier": "tek@fazier.test", "filled_fields": 2}
        assert meta.pending and meta.generated and meta.origin == FAZIER
        assert "Never ask the person to type a password in chat" in out["next"]
        _no_secret_anywhere(password, raw, caplog.text, page.plain_exprs, spawned)
        from agent.redact import redact_registered_vault_values
        assert password not in redact_registered_vault_values(f"echo {password}")  # registered with the egress boundary

    def test_a_fill_that_throws_leaks_nothing_and_leaves_no_orphan(self, store, caplog):
        from tools import browser_vault_tool as bvt

        caplog.set_level(logging.DEBUG)
        page = Page(fill_error=lambda expr: RuntimeError("renderer crashed while running " + expr))
        with _live(True):
            raw = _run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER, username="tek@fazier.test",
                       generate=True)
        password = page.fills()[0]["value"]
        out = json.loads(raw)
        assert out["success"] is False and store.list_items() == []
        _no_secret_anywhere(password, raw, caplog.text)

    def test_not_a_signup_form_creates_nothing(self, store):
        from tools import browser_vault_tool as bvt

        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        with _live(True):
            out = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER, username="a@b.test",
                                  generate=True))
        assert out["outcome"] == "not_a_signup_form" and store.list_items() == [] and page.secret_exprs == []

    def test_generate_needs_url_and_username(self, store):
        from tools import browser_vault_tool as bvt

        out = json.loads(_run(Page(), bvt.browser_vault_save_login, task_id="t", generate=True))
        assert out["error_type"] == "missing_argument" and store.list_items() == []

    def test_the_login_entry_point_generates_on_signup(self, store):
        from tools import browser_vault_tool as bvt

        page = Page()
        with _live(True):
            out = json.loads(_run(page, bvt.browser_vault_login, FAZIER + "/signup", mode="signup",
                                  username="tek@fazier.test", task_id="t"))
        assert out["outcome"] == "created" and out["pending"] and len(page.fills()) == 2


class TestSignupClassifier:
    def _c(self, rows):
        from agent.vault_login_classifier import LoginControl, classify_signup_controls
        return [c.control.index for c in classify_signup_controls([LoginControl.from_dict(r) for r in rows])]

    def test_new_and_confirm_are_both_targets(self):
        assert self._c(SIGNUP_CONTROLS) == [1, 2]

    def test_a_login_form_is_not_a_signup_form(self):
        assert self._c(LOGIN_CONTROLS) == []

    def test_bare_two_box_form_and_mixed_page(self):
        bare = [{"index": 3, "type": "password", "name": "pw1", "label": "", "autocomplete": "", "formIndex": 1},
                {"index": 4, "type": "password", "name": "pw2", "label": "", "autocomplete": "", "formIndex": 1}]
        assert self._c(bare) == [3, 4]
        mixed = LOGIN_CONTROLS + [{"index": 5, "type": "password", "name": "newPassword", "label": "", "autocomplete": "",
                                   "formIndex": 1}]
        assert self._c(mixed) == [5]  # the sign-in box on the same page is left alone

    def test_the_login_classifier_still_excludes_signup_fields(self):
        from agent.vault_login_classifier import LoginControl, classify_login_control
        assert classify_login_control(LoginControl.from_dict(SIGNUP_CONTROLS[1])) is None


class TestPasswordRules:
    def test_default_is_twenty_with_every_class(self):
        from agent import vault_password as vp
        pw = vp.generate()
        assert len(pw) == 20 and all(re.search(p, pw) for p in (r"[A-Z]", r"[a-z]", r"\d", r"[!@#$%*\-_.?]"))

    def test_apple_passwordrules_on_the_page_are_obeyed(self):
        from agent import vault_password as vp
        from agent.vault_login_classifier import LoginControl
        policy = vp.policy_from_controls([LoginControl.from_dict(SIGNUP_CONTROLS[1])])
        assert policy.symbols == "-_" and set(policy.required) == {"upper", "digit"}
        for _ in range(50):
            pw = vp.generate(policy)
            assert 12 <= len(pw) <= 16 and re.search(r"[A-Z]", pw) and re.search(r"\d", pw)
            assert set(re.sub(r"[A-Za-z0-9]", "", pw)) <= {"-", "_"}

    def test_pattern_without_symbols_and_a_max(self):
        from agent import vault_password as vp
        from agent.vault_login_classifier import LoginControl
        policy = vp.policy_from_controls([LoginControl.from_dict({"index": 0, "type": "password", "maxLength": 12,
                                                                  "pattern": "[A-Za-z0-9]{8,12}"})])
        for _ in range(20):
            pw = vp.generate(policy)
            assert re.fullmatch(r"[A-Za-z0-9]{8,12}", pw)

    def test_request_rules(self):
        from agent import vault_password as vp
        policy = vp.policy_from_request(vp.PasswordPolicy(), {"max_length": 10, "symbols": False})
        pw = vp.generate(policy)
        assert len(pw) == 10 and pw.isalnum()
        policy = vp.policy_from_request(vp.PasswordPolicy(), {"allowed_symbols": "#", "require": ["symbol", "digit"]})
        pw = vp.generate(policy)
        assert "#" in pw and set(re.sub(r"[A-Za-z0-9]", "", pw)) == {"#"}


# ---------------------------------------------------------------------------
# 4. pending: confirm, regenerate, prune
# ---------------------------------------------------------------------------

class TestPending:
    def _created(self, store):
        from tools import browser_vault_tool as bvt
        page = Page()
        with _live(True):
            out = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER, username="tek@fazier.test",
                                  generate=True))
        return out["handle"], page

    def test_regenerate_replaces_only_while_pending_then_confirm_is_final(self, store):
        from tools import browser_vault_tool as bvt

        handle, _ = self._created(store)
        first = store.resolve_secret(handle)["password"]
        page = Page()
        with _live(True):
            raw = _run(page, bvt.browser_vault_regenerate, handle, policy={"max_length": 12, "symbols": False}, task_id="t")
        out = json.loads(raw)
        second = store.resolve_secret(handle)["password"]
        assert out["outcome"] == "created" and out["handle"] == handle and len(store.list_items()) == 1
        assert second != first and len(second) == 12 and second.isalnum()
        assert {f["value"] for f in page.fills()} == {second} and second not in raw

        confirmed = json.loads(bvt.browser_vault_confirm(handle))
        assert confirmed == {"success": True, "handle": handle, "origin": FAZIER, "pending": False}
        assert store.get_meta(handle).pending is False
        with _live(True):
            refused = json.loads(_run(Page(), bvt.browser_vault_regenerate, handle, task_id="t"))
        assert refused["error_type"] == "not_pending" and store.resolve_secret(handle)["password"] == second

    def test_a_signup_the_site_refused_is_discarded_and_the_created_result_says_so(self, store):
        """2026-09-29 fazier: "Email has already been taken" -- the pending login for an account that was never
        made stayed in the vault. The created result tells the model to discard it; discard removes only a
        pending login."""
        from tools import browser_vault_tool as bvt
        page = Page()
        with _live(True):
            created = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER,
                                      username="tek@fazier.test", generate=True))
        assert "browser_vault_discard(handle)" in created["next"] and "already been taken" in created["next"]
        handle = created["handle"]
        out = json.loads(bvt.browser_vault_discard(handle))
        assert out == {"success": True, "handle": handle, "origin": FAZIER, "discarded": True}
        assert store.get_meta(handle) is None and store.list_items() == []
        assert json.loads(bvt.browser_vault_discard(handle))["error_type"] == "unknown_handle"

    def test_discard_never_removes_a_confirmed_login(self, store):
        from tools import browser_vault_tool as bvt
        meta = store.add_item("login", "f", {"identifier_type": "email", "identifier": "a@b.test", "password": "keep-1"},
                              origin=FAZIER, pending=True, generated=True)
        store.confirm_item(meta.id)
        out = json.loads(bvt.browser_vault_discard(meta.id))
        assert out["error_type"] == "not_pending" and store.resolve_secret(meta.id)["password"] == "keep-1"
        assert store.discard_pending(meta.id) is False and store.get_meta(meta.id) is not None

    def test_the_store_never_rewrites_a_confirmed_login(self, store):
        meta = store.add_item("login", "f", {"identifier_type": "email", "identifier": "a@b.test", "password": "keep-1"},
                              origin=FAZIER, pending=True, generated=True)
        assert store.replace_pending_password(meta.id, "second-2").pending is True
        assert store.confirm_item(meta.id).pending is False and store.confirm_item(meta.id) is None
        assert store.replace_pending_password(meta.id, "third-3") is None
        assert store.resolve_secret(meta.id)["password"] == "second-2"

    def test_a_retried_signup_reuses_its_pending_login(self, store):
        handle, _ = self._created(store)
        again, _ = self._created(store)
        assert again == handle and len(store.list_items()) == 1

    def test_stale_pending_logins_are_pruned_and_confirmed_ones_kept(self, store):
        from agent.vault_store import PENDING_MAX_AGE_S
        from tools import browser_vault_tool as bvt

        stale = store.add_item("login", "old", {"identifier_type": "email", "identifier": "a@b.test", "password": "p1"},
                               origin="https://old.test", pending=True, generated=True)
        kept = store.add_item("login", "kept", {"identifier_type": "email", "identifier": "a@b.test", "password": "p2"},
                              origin="https://kept.test")
        fresh = store.add_item("login", "new", {"identifier_type": "email", "identifier": "a@b.test", "password": "p3"},
                               origin="https://new.test", pending=True, generated=True)
        assert store.prune_pending(now=time.time() + 60) == 0
        with patch("time.time", return_value=time.time() + PENDING_MAX_AGE_S + 60):
            store.confirm_item(fresh.id)  # confirmed in time: survives
            listed = json.loads(bvt.browser_vault_list())
        assert {i["handle"] for i in listed["items"]} == {kept.id, fresh.id}
        assert store.get_meta(stale.id) is None


# ---------------------------------------------------------------------------
# 5. the host's prompt program
# ---------------------------------------------------------------------------

def _program(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return str(path)


OK_LOGIN = """
import json, os, sys
req = json.loads(sys.stdin.read())
with open(os.environ["FAKE_PROMPT_SEEN"], "w") as fh:
    json.dump({"request": req, "argv": sys.argv, "env_has_secret": any("pw-Typed-7" in v for v in os.environ.values())}, fh)
print(json.dumps({"ok": True, "username": "tek@fazier.test", "password": "pw-Typed-7"}))
"""


class TestPromptProgram:
    def test_login_contract(self, tmp_path, monkeypatch, caplog):
        from agent.vault_backends import secret_prompt as sp

        seen = tmp_path / "seen.json"
        monkeypatch.setenv("FAKE_PROMPT_SEEN", str(seen))
        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "ok.py", OK_LOGIN))
        caplog.set_level(logging.DEBUG)
        answer = sp.run_prompt("login", origin=FAZIER, label="fazier.com", username="tek@fazier.test", message="m")
        assert answer == {"ok": True, "username": "tek@fazier.test", "password": "pw-Typed-7"}
        got = json.loads(seen.read_text())
        assert got["request"] == {"kind": "login", "origin": FAZIER, "label": "fazier.com", "username": "tek@fazier.test",
                                  "message": "m", "fields": ["username", "password"]}
        assert len(got["argv"]) == 1 and got["env_has_secret"] is False
        assert "pw-Typed-7" not in caplog.text

    @pytest.mark.parametrize("body,reason", [
        ('import sys; sys.stdin.read(); print(\'{"ok": false, "reason": "cancelled"}\'); sys.exit(1)', "cancelled"),
        ('import sys; sys.stdin.read(); print("hello pw-Leak-1 not json")', "bad_output"),
        ('import sys; sys.stdin.read(); print(\'{"ok": true, "username": "u", "password": "pw-Leak-1"}\'); sys.exit(1)', "bad_output"),
        ('import sys; sys.stdin.read(); print(\'{"ok": true, "username": "u"}\')', "bad_output"),
        ('import sys; sys.stdin.read(); print(\'[1, 2]\')', "bad_output"),
    ])
    def test_refusals(self, tmp_path, monkeypatch, caplog, body, reason):
        from agent.vault_backends import secret_prompt as sp

        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "p.py", body + "\n"))
        caplog.set_level(logging.DEBUG)
        assert sp.run_prompt("login", origin=FAZIER) == {"ok": False, "reason": reason}
        assert "pw-Leak-1" not in caplog.text

    def test_timeout(self, tmp_path, monkeypatch):
        from agent.vault_backends import secret_prompt as sp

        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "slow.py", "import time; time.sleep(30)\n"))
        started = time.monotonic()
        assert sp.run_prompt("code", origin=FAZIER, timeout=1) == {"ok": False, "reason": "timeout"}
        assert time.monotonic() - started < 10
        assert sp.TIMEOUT_S == 300

    def test_a_relative_or_missing_program_is_never_run(self, tmp_path, monkeypatch):
        from agent.vault_backends import secret_prompt as sp

        monkeypatch.setenv(sp.ENV_CMD, "ok.py")
        assert sp.prompt_command() is None and sp.run_prompt("login") == {"ok": False, "reason": "unavailable"}
        monkeypatch.setenv(sp.ENV_CMD, str(tmp_path / "nope"))
        assert sp.prompt_command() is None

    def _mcp_like(self, monkeypatch, tmp_path, *, here):
        """The hermes-tools MCP server: headless by env, no in-process prompt, the host's program set."""
        from agent.vault_backends import secret_prompt as sp

        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        seen = tmp_path / "seen.json"
        monkeypatch.setenv("FAKE_PROMPT_SEEN", str(seen))
        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "ok.py", OK_LOGIN))
        return seen, patch("tools.browser_chrome_extension.at_this_mac",
                           return_value=(True, "the person's own turn, at the Mac") if here else (False, "a turn from the person's phone"))

    def test_save_login_uses_the_program_when_the_person_is_at_the_mac(self, store, tmp_path, monkeypatch, caplog):
        from tools import browser_vault_tool as bvt

        seen, here = self._mcp_like(monkeypatch, tmp_path, here=True)
        caplog.set_level(logging.DEBUG)
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        with here, _live(True):
            raw = _run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER, username="tek@fazier.test")
        out = json.loads(raw)
        [meta] = store.list_items()
        assert out["outcome"] == "saved" and meta.origin == FAZIER and meta.identifier == "tek@fazier.test"
        assert store.resolve_secret(meta.id)["password"] == "pw-Typed-7"
        assert json.loads(seen.read_text())["request"]["username"] == "tek@fazier.test"
        assert page.fills() == [{"index": 1, "token": "current-password", "value": "pw-Typed-7"}]
        _no_secret_anywhere("pw-Typed-7", raw, caplog.text, page.plain_exprs)

    def test_not_at_the_mac_the_program_is_not_run(self, store, tmp_path, monkeypatch):
        from tools import browser_vault_tool as bvt

        seen, away = self._mcp_like(monkeypatch, tmp_path, here=False)
        page = Page(url=FAZIER + "/login", controls=LOGIN_CONTROLS)
        with away, _live(True):
            live = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER))
        with away, _live(False):
            cron = json.loads(_run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER))
        assert live["error_type"] == "prompt_unavailable" and "phone" in live["error"]
        assert "Desktop" not in live["error"] and "hermes vault add" not in live["error"]
        assert cron["code"] == "person_needed" and cron["outcome"] == "needs_person"
        assert not seen.exists() and store.list_items() == []

    def test_code_and_unlock_prompts_through_the_program(self, tmp_path, monkeypatch):
        from agent.vault_backends import unlock as unlock_mod
        from agent.vault_backends import secret_prompt as sp
        from tools import browser_vault_tool as bvt

        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        body = ("import json, sys; r = json.loads(sys.stdin.read()); "
                "print(json.dumps({'ok': True, 'code': '482913'} if r['kind'] == 'code' else {'ok': True, 'password': 'master-pw'}))\n")
        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "p.py", body))
        at_mac = patch("tools.browser_chrome_extension.at_this_mac", return_value=(True, "here"))
        controls = [{"index": 0, "type": "text", "name": "otp", "label": "Code", "autocomplete": "one-time-code"}]
        page = Page(url=FAZIER + "/2fa", controls=controls)
        with at_mac, _live(True), patch("agent.vault_code_sources.registered_sources", return_value=[]):
            out = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t"))
            unlock, _why = unlock_mod.resolve_unlock_prompt()
            assert unlock("bitwarden", "Bitwarden") == "master-pw"
        assert out["success"] and out["source"] == "user" and page.fills()[0]["value"] == "482913"
        assert "482913" not in json.dumps(out)


# ---------------------------------------------------------------------------
# 6. nobody here: nothing created
# ---------------------------------------------------------------------------

class TestUnattended:
    @pytest.mark.parametrize("call", ["save_generate", "login_signup", "regenerate"])
    def test_person_needed_and_nothing_stored(self, store, call):
        from tools import browser_vault_tool as bvt

        page = Page()
        if call == "regenerate":
            meta = store.add_item("login", "f", {"identifier_type": "email", "identifier": "a@b.test", "password": "p-1"},
                                  origin=FAZIER, pending=True, generated=True)
        with _live(False):
            if call == "save_generate":
                raw = _run(page, bvt.browser_vault_save_login, task_id="t", url=FAZIER, username="a@b.test", generate=True)
            elif call == "login_signup":
                raw = _run(page, bvt.browser_vault_login, FAZIER, mode="signup", username="a@b.test", task_id="t")
            else:
                raw = _run(page, bvt.browser_vault_regenerate, meta.id, task_id="t")
        out = json.loads(raw)
        assert out["code"] == "person_needed" and out["outcome"] == "needs_person" and out["retry"] is False
        assert page.secret_exprs == []
        if call == "regenerate":
            assert store.resolve_secret(meta.id)["password"] == "p-1"
        else:
            assert store.list_items() == []


def test_new_tools_are_registered_in_the_browser_toolset():
    import toolsets
    from tools import browser_vault_tool  # noqa: F401
    from tools.registry import registry

    names = {e.name for e in registry.get_all_entries()}
    for name in ("browser_vault_login", "browser_vault_confirm", "browser_vault_regenerate", "browser_vault_discard"):
        assert name in names and name in toolsets.TOOLSETS["browser"]["tools"]


@pytest.mark.parametrize("said, meant", [("new", "signup"), ("register", "signup"), ("create", "signup"),
                                         ("Sign-Up", "signup"), ("create_account", "signup"),
                                         ("signin", "login"), ("sign_in", "login"), ("Log-In", "login")])
def test_the_modes_models_say_are_understood(store, said, meant):
    """2026-09-29: the owner approved browser_vault_login(mode="new") in Moe's dialog, and it then failed
    bad_mode -- an approval spent on an argument the tool could have understood."""
    from tools import browser_vault_tool as bvt
    seen = {}
    with patch.object(bvt, "_bind_page", lambda task, kind, url: seen.update(bound=True) or (None, json.dumps({"stop": 1}))):
        out = json.loads(bvt.browser_vault_login(FAZIER, mode=said, username="a@b.test", task_id="t"))
    assert out == {"stop": 1} and seen["bound"], said               # got past the mode check
    assert bvt._MODE_SYNONYMS[re.sub(r"[\s_-]+", "", said.lower())] == meant


def test_an_unknown_mode_is_still_refused_and_the_schema_says_what_is_required():
    from tools import browser_vault_tool as bvt
    out = json.loads(bvt.browser_vault_login(FAZIER, mode="maybe", task_id="t"))
    assert out["error_type"] == "bad_mode"
    schema = bvt.BROWSER_VAULT_LOGIN_SCHEMA
    assert "REQUIRED when mode is 'signup'" in schema["parameters"]["properties"]["username"]["description"]
    assert "username too when mode is 'signup'" in schema["description"]


# ---------------------------------------------------------------------------
# 7. a passcode / PIN the person knows (X's "Enter Passcode" for encrypted chats, 2026-09-29)
# ---------------------------------------------------------------------------

X = "https://x.com"
#: What X showed: four bare <input>s -- no name, no id, no label -- under "Enter Passcode".
X_BOXES = [{"index": i, "type": "text", "name": " ", "label": "    ", "autocomplete": "", "formIndex": None}
           for i in range(4)]
X_WORDS = "X / Chat \n Enter Passcode Enter your passcode to access your encrypted messages. Forgot passcode?"
PASSCODE_PROGRAM = """
import json, os, sys
req = json.loads(sys.stdin.read())
with open(os.environ["FAKE_PROMPT_SEEN"], "w") as fh:
    json.dump({"request": req, "argv": sys.argv}, fh)
print(json.dumps({"ok": True, "code": "__PASSCODE__"}))
"""


class WordsPage(Page):
    def __init__(self, words=X_WORDS, **kw):
        super().__init__(**kw)
        self.words = words

    def eval(self, task_id, expr):
        if "document.body.innerText" in expr:
            self.plain_exprs.append(expr)
            return {"success": True, "result": self.words}
        return super().eval(task_id, expr)


class TestPasscode:
    def _program(self, monkeypatch, tmp_path, *, here=True, passcode="4821"):
        from agent.vault_backends import secret_prompt as sp
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        seen = tmp_path / "seen.json"
        monkeypatch.setenv("FAKE_PROMPT_SEEN", str(seen))
        # the value lives in the program, never in the environment (_no_secret_anywhere checks it)
        monkeypatch.setenv(sp.ENV_CMD, _program(tmp_path, "pc.py", PASSCODE_PROGRAM.replace("__PASSCODE__", passcode)))
        return seen, patch("tools.browser_chrome_extension.at_this_mac",
                           return_value=(True, "here") if here else (False, "a turn from the person's phone"))

    @staticmethod
    def _no_mail():
        return patch("agent.vault_code_sources.find_code", side_effect=AssertionError("the mail was read"))

    def test_the_x_boxes_are_a_split_passcode_and_were_invisible_to_the_code_reader(self):
        from agent.vault_login_classifier import LoginControl, classify_otp_controls, classify_passcode_controls
        controls = [LoginControl.from_dict(c) for c in X_BOXES]
        assert classify_otp_controls(controls) == []                 # the gap: nothing to type into
        chosen, layout = classify_passcode_controls(controls, X_WORDS)
        assert layout == "split" and [c.control.index for c in chosen] == [0, 1, 2, 3]
        assert classify_passcode_controls(controls, "Welcome to X") == ([], "")   # bare boxes alone are not a passcode

    @pytest.mark.parametrize("control, found", [
        ({"type": "password", "name": "pin", "label": "", "maxLength": 6}, True),          # a password-type PIN box
        ({"type": "tel", "name": "x1", "label": "Security code", "maxLength": 8}, True),
        ({"type": "text", "name": "q", "label": "", "context": "Enter your passcode"}, True),
        ({"type": "password", "name": "password", "label": "Password", "autocomplete": "current-password"}, False),
        ({"type": "email", "name": "email", "label": "Email"}, False),
        ({"type": "text", "name": "promo", "label": "Promo code"}, False),
    ])
    def test_single_fields(self, control, found):
        from agent.vault_login_classifier import LoginControl, classify_passcode_controls
        c = LoginControl.from_dict({"index": 3, "autocomplete": "", "formIndex": 0, **control})
        chosen, layout = classify_passcode_controls([c], "")
        assert bool(chosen) is found and (layout == "single" if found else layout == "")

    def test_split_password_type_digit_boxes(self):
        from agent.vault_login_classifier import LoginControl, classify_passcode_controls
        boxes = [LoginControl.from_dict({"index": 5 + i, "type": "password", "name": f"d{i}", "label": "",
                                         "maxLength": 1, "formIndex": 1}) for i in range(6)]
        chosen, layout = classify_passcode_controls(boxes, "Enter your 6-digit PIN")
        assert layout == "split" and len(chosen) == 6

    def test_x_passcode_goes_from_moe_s_window_into_the_four_boxes(self, store, tmp_path, monkeypatch, caplog):
        from tools import browser_vault_tool as bvt
        caplog.set_level(logging.DEBUG)
        seen, here = self._program(monkeypatch, tmp_path)
        page = WordsPage(url=X + "/i/chat/pin/recovery", controls=X_BOXES)
        with here, _live(True), self._no_mail():
            raw = _run(page, bvt.browser_vault_enter_code, task_id="t", url=X, kind="passcode")
        out = json.loads(raw)
        assert out["success"] and out["kind"] == "passcode" and out["layout"] == "split" and out["source"] == "person_prompt"
        assert page.fills() == [{"index": i, "token": "passcode", "value": ch} for i, ch in enumerate("4821")]
        request = json.loads(seen.read_text())["request"]
        assert request == {"kind": "code", "origin": "https://x.com", "label": "x.com",
                           "message": "Enter your passcode for x.com (4 digits). Moe types it into the page without seeing it.",
                           "fields": ["code"]}
        assert "remembered" not in out and store.list_items() == []
        _no_secret_anywhere("4821", raw, caplog.text, page.plain_exprs)

    def test_code_mode_on_a_passcode_page_redirects_without_reading_mail_or_typing(self, tmp_path, monkeypatch):
        from tools import browser_vault_tool as bvt
        page = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with _live(True), self._no_mail(), patch("agent.vault_code_sources.registered_sources", return_value=["gmail"]):
            out = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t", url=X))
        assert out["error_type"] == "passcode_page" and 'kind=\"passcode\"' in out["error"]
        assert page.secret_exprs == []

    def test_code_mode_on_a_sent_code_page_is_unchanged(self, monkeypatch):
        from tools import browser_vault_tool as bvt
        controls = [{"index": 0, "type": "text", "name": "otp", "label": "Code", "autocomplete": "one-time-code"}]
        page = WordsPage(url=FAZIER + "/2fa", controls=controls, words="We sent a 6-digit code to your email. Enter it below.")
        with _live(True), patch("agent.vault_code_sources.registered_sources", return_value=["gmail"]), \
                patch("agent.vault_code_sources.find_code", return_value=("482913", "gmail")):
            out = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t"))
        assert out["success"] and out["source"].startswith("email")

    def test_remember_keeps_it_with_the_saved_login_and_next_time_nobody_is_asked(self, store, tmp_path, monkeypatch, caplog):
        from tools import browser_vault_tool as bvt
        from agent.vault_backends import secret_prompt as sp
        caplog.set_level(logging.DEBUG)
        meta = store.add_item("login", "x", {"identifier_type": "username", "identifier": "izzy", "password": "pw-1"},
                              origin=X)
        seen, here = self._program(monkeypatch, tmp_path)
        page = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with here, _live(True), self._no_mail():
            first = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t", kind="passcode", remember=True))
        assert first["remembered"] is True and first["handle"] == meta.id
        assert store.get_meta(meta.id).has_passcode and store.resolve_secret(meta.id)["password"] == "pw-1"
        seen.unlink()
        monkeypatch.delenv(sp.ENV_CMD)                      # no window at all now
        page2 = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with _live(True), self._no_mail():
            raw = _run(page2, bvt.browser_vault_enter_code, task_id="t", kind="passcode")
        again = json.loads(raw)
        assert again["success"] and again["source"] == "saved" and not seen.exists()
        assert [f["value"] for f in page2.fills()] == list("4821")
        _no_secret_anywhere("4821", json.dumps(first), raw, caplog.text, page.plain_exprs, page2.plain_exprs)

    def test_remember_without_a_saved_login_is_used_once_and_not_stored(self, store, tmp_path, monkeypatch):
        from tools import browser_vault_tool as bvt
        seen, here = self._program(monkeypatch, tmp_path)
        page = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with here, _live(True), self._no_mail():
            out = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t", kind="passcode", remember=True))
        assert out["success"] and out["remembered"] is False and "no login" in out["why"] and store.list_items() == []

    def test_wrong_length_types_nothing_and_never_says_the_value(self, tmp_path, monkeypatch):
        from tools import browser_vault_tool as bvt
        seen, here = self._program(monkeypatch, tmp_path, passcode="482177")
        page = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with here, _live(True), self._no_mail():
            raw = _run(page, bvt.browser_vault_enter_code, task_id="t", kind="passcode")
        out = json.loads(raw)
        assert out["error_type"] == "passcode_length" and page.secret_exprs == [] and "482177" not in raw

    def test_nobody_at_the_mac_opens_no_window(self, tmp_path, monkeypatch):
        from tools import browser_vault_tool as bvt
        seen, away = self._program(monkeypatch, tmp_path, here=False)
        page = WordsPage(url=X + "/i/chat", controls=X_BOXES)
        with away, _live(False), self._no_mail():
            cron = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t", kind="passcode"))
        with away, _live(True), self._no_mail():
            phone = json.loads(_run(page, bvt.browser_vault_enter_code, task_id="t", kind="passcode"))
        assert cron["code"] == "person_needed" and phone["error_type"] == "prompt_unavailable"
        assert "browser_handoff" in phone["error"] and "chat" in phone["error"]
        assert not seen.exists() and page.secret_exprs == []

    def test_the_schema_says_never_refuse_and_never_in_chat(self):
        from tools import browser_vault_tool as bvt
        s = bvt.BROWSER_VAULT_ENTER_CODE_SCHEMA
        assert "never refuse and never ask them to type it in chat" in s["description"]
        assert s["parameters"]["properties"]["kind"]["enum"] == ["passcode"]
        assert s["parameters"]["properties"]["remember"]["default"] is False


def _cft():
    for app in sorted((Path.home() / ".agent-browser" / "browsers").glob("chrome-*/Google Chrome for Testing.app"), reverse=True):
        exe = app / "Contents" / "MacOS" / "Google Chrome for Testing"
        if exe.exists():
            return str(exe)
    return None


@pytest.mark.skipif(_cft() is None, reason="Chrome for Testing not installed under ~/.agent-browser/browsers")
def test_live_passcode_boxes_are_found_and_filled_in_a_real_page(tmp_path):
    """X's shape on a local page in Chrome for Testing: four bare inputs (two of them type=password) that
    move focus on input, under "Enter Passcode". Found by the real inspection script, filled one digit each."""
    import base64
    import signal
    from agent.vault_login_classifier import (LoginControl, build_fill_js, build_inspection_js,
                                              build_passcode_fills, classify_passcode_controls)
    from tools import browser_captcha_ladder as bl
    from tools import browser_vault_tool as bvt
    html = """<html><body><h1>Enter Passcode</h1><div>
<input type="text" style="width:2em"><input type="password" style="width:2em"><input type="password" style="width:2em"><input type="text" style="width:2em">
</div><script>
window.__got = [];
document.querySelectorAll('input').forEach((el, i, all) => el.addEventListener('input', () => {
  window.__got.push(i); if (all[i + 1]) all[i + 1].focus(); }));
</script></body></html>"""
    url = "data:text/html;base64," + base64.b64encode(html.encode()).decode()
    ud = tmp_path / "ud"
    proc = subprocess.Popen([_cft(), "--headless=new", f"--user-data-dir={ud}", "--remote-debugging-port=0",
                             "--no-first-run", "--no-proxy-server", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        port_file = ud / "DevToolsActivePort"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not (port_file.exists() and port_file.read_text().strip()):
            time.sleep(0.3)
        if not (port_file.exists() and port_file.read_text().strip()):
            pytest.skip("Chrome for Testing did not start in 90 s (a loaded Mac is not a result)")
        port, path = port_file.read_text().split("\n")[:2]
        with bl.CdpConn(f"ws://127.0.0.1:{port}{path}", timeout=30) as conn:
            tid = conn.call("Target.createTarget", {"url": "about:blank"})["targetId"]
            sid = conn.call("Target.attachToTarget", {"targetId": tid, "flatten": True})["sessionId"]
            conn.call("Page.navigate", {"url": url}, session_id=sid)

            def ev(expr):
                r = conn.call("Runtime.evaluate", {"expression": expr, "returnByValue": True}, session_id=sid)
                return (r.get("result") or {}).get("value")
            for _ in range(100):
                if ev("document.readyState + document.querySelectorAll('input').length") == "complete4":
                    break
                time.sleep(0.2)
            controls = [LoginControl.from_dict(c) for c in json.loads(ev(build_inspection_js("n1")))]
            chosen, layout = classify_passcode_controls(controls, ev(bvt._PAGE_WORDS_JS))
            assert layout == "split" and [c.control.index for c in chosen] == [0, 1, 2, 3]
            out = json.loads(ev(build_fill_js(build_passcode_fills(chosen, layout, "4821"), expected_origin="null",
                                              nonce="n1")))
            assert out == {"filled": 4}
            assert ev("Array.from(document.querySelectorAll('input'), e => e.value).join('')") == "4821"
            assert ev("JSON.stringify(window.__got)") == "[0,1,2,3]"
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
