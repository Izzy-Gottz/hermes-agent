"""An agent built with skip_background_review (cron) spawns no review on any runtime — the claude_code
and codex runtimes called _spawn_background_review without checking it (Memoe, 2026-09-30)."""
import types

import run_agent


def test_skip_holds_for_every_caller():
    spawned = []
    fake = types.SimpleNamespace(skip_background_review=True, _delegate_depth=0,
                                 _spawn_background_review_now=lambda **kw: spawned.append(kw))
    run_agent.AIAgent._spawn_background_review(fake, messages_snapshot=[], review_memory=True)
    assert spawned == []
    # /refine (explicit) still runs: it is the person asking.
    fake.skip_background_review = True
    try:
        run_agent.AIAgent._spawn_background_review(fake, messages_snapshot=[], review_memory=True, explicit=True)
    except Exception:
        pass
