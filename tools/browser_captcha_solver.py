"""Tier C of the CAPTCHA ladder: the seam for a recognition solver. Nothing here reaches a network.

A recognition solver answers one narrow question -- *which tiles of this image grid match the
instruction* (or *what does this image say*) -- and the ladder does every click itself, in the owner's
own browser, from the owner's own IP. The solver never sees a cookie, a URL beyond the image, or the
page; it gets pixels and an instruction and returns indices.

Bring your own key: a provider (CapSolver, NopeCHA, a local model) is registered with
:func:`register_solver` and chosen by ``browser.captcha.solver.provider``. None ships in this slice, so
:func:`solver_for` returns None and the ladder hands the image challenge to the person.

Two refusals are structural, not configurable:

* Cloudflare kinds (``NEVER_THIRD_PARTY`` in tools/browser_captcha.py) never get a solver -- the owner
  ruled out sending Cloudflare challenge / ``cf_clearance`` work to a third party.
* No solver is ever asked for audio or an accessibility route; :class:`SolveRequest` has no field that
  could carry one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from tools.browser_captcha import NEVER_THIRD_PARTY, captcha_config

#: Kinds a recognition solver may be asked about (image grids and text images). Everything else is a
#: click / hold (tier A) or the person.
SOLVABLE_KINDS = frozenset({"recaptcha_v2", "hcaptcha", "image_text", "geetest", "arkose", "aws_waf"})


@dataclass(frozen=True)
class SolveRequest:
    kind: str
    image_png: bytes
    instruction: str                 # the challenge's own words, e.g. "Select all images with buses"
    grid: Tuple[int, int] = (3, 3)   # rows, cols; (1, 1) for a single text image


@dataclass(frozen=True)
class SolveAnswer:
    tiles: Tuple[int, ...] = ()      # 0-based, row-major
    text: str = ""                   # for image_text
    confidence: Optional[float] = None


class RecognitionSolver(Protocol):
    name: str

    def supports(self, kind: str) -> bool: ...

    def solve(self, request: SolveRequest) -> SolveAnswer: ...


_REGISTRY: Dict[str, Callable[[dict], RecognitionSolver]] = {}


def register_solver(provider: str, factory: Callable[[dict], RecognitionSolver]) -> None:
    """Make ``provider`` choosable by ``browser.captcha.solver.provider``; ``factory(solver_cfg)`` builds it
    (reading its own key, e.g. from the env var the config names)."""
    _REGISTRY[str(provider).strip().lower()] = factory


def unregister_solver(provider: str) -> None:
    _REGISTRY.pop(str(provider).strip().lower(), None)


def registered() -> Sequence[str]:
    return tuple(sorted(_REGISTRY))


def solver_for(kind: str, browser_cfg: Optional[dict]) -> Tuple[Optional[RecognitionSolver], str]:
    """``(solver, why)`` -- a solver for ``kind``, or None and the reason there is none."""
    if kind in NEVER_THIRD_PARTY:
        return None, "Cloudflare challenges are never sent to a third party"
    if kind not in SOLVABLE_KINDS:
        return None, f"{kind} is not a recognition task"
    cfg = captcha_config(browser_cfg).get("solver")
    provider = str((cfg or {}).get("provider") or "").strip().lower() if isinstance(cfg, dict) else ""
    if not provider:
        return None, "no recognition solver is configured (browser.captcha.solver.provider)"
    factory = _REGISTRY.get(provider)
    if factory is None:
        return None, f"recognition solver {provider!r} is not installed"
    try:
        solver = factory(dict(cfg))
    except Exception as exc:
        return None, f"recognition solver {provider!r} could not start: {exc}"
    if not solver.supports(kind):
        return None, f"recognition solver {provider!r} does not handle {kind}"
    return solver, provider


def validate_answer(answer: SolveAnswer, grid: Tuple[int, int]) -> List[int]:
    """The tile indices, in range and de-duplicated, or ValueError -- a solver's output is data, never trusted."""
    n = max(1, grid[0]) * max(1, grid[1])
    out: List[int] = []
    for t in answer.tiles:
        if not isinstance(t, int) or not 0 <= t < n:
            raise ValueError(f"tile {t!r} outside a {grid[0]}x{grid[1]} grid")
        if t not in out:
            out.append(t)
    return out
