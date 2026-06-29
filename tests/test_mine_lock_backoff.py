"""Tests for the quiet bounded mine-lock retry (FIX 2).

A direct ``mempalace mine`` against a palace a daemon/concurrent mine holds
used to fail fast and, across a harvest, print the same
"held by PID … wait for it to finish" line over and over. ``cli._mine_with_lock_backoff``
now collapses that contention into ONE notice + an exponential backoff,
bounds the total wait, then either runs (lock frees) or gives up cleanly.

These unit-test the helper directly with a virtual clock — no real sleeps,
no real palace, no subprocess — so they are fast and deterministic.
"""

from __future__ import annotations

import pytest

from mempalace import cli
from mempalace.palace import MineAlreadyRunning


@pytest.fixture
def virtual_clock(monkeypatch):
    """Replace ``cli.time`` clock + sleep with a fake advancing clock."""
    state = {"t": 0.0, "sleeps": []}

    def fake_monotonic():
        return state["t"]

    def fake_sleep(seconds):
        state["sleeps"].append(seconds)
        state["t"] += seconds

    monkeypatch.setattr(cli.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    return state


def test_retries_quietly_then_succeeds(virtual_clock, capsys):
    """Contention that clears within budget → one notice, then success."""
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        if calls["n"] < 3:
            raise MineAlreadyRunning("palace /p is held by PID 999 (mempalace-mcp)")
        return "mined"

    result = cli._mine_with_lock_backoff(attempt, max_wait=30.0)

    assert result == "mined"
    assert calls["n"] == 3
    # Backed off twice (before the 2nd and 3rd attempts), exponentially.
    assert virtual_clock["sleeps"] == [0.5, 1.0]

    err = capsys.readouterr().err
    # The holder line appears exactly ONCE — not once per failed attempt.
    assert err.count("held by PID 999") == 1
    assert "quiet retry with backoff" in err


def test_gives_up_cleanly_after_budget(virtual_clock, capsys):
    """Contention that never clears → bounded wait, clean SystemExit(1)."""
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        raise MineAlreadyRunning("palace /p is held by PID 999 (mempalace-mcp)")

    with pytest.raises(SystemExit) as exc_info:
        cli._mine_with_lock_backoff(attempt, max_wait=10.0)

    assert exc_info.value.code == 1
    # Wait is bounded: total backoff never exceeds the budget.
    assert sum(virtual_clock["sleeps"]) <= 10.0
    # It retried a handful of times, not dozens — and certainly not forever.
    assert 1 < calls["n"] < 20

    err = capsys.readouterr().err
    assert "giving up" in err
    # Still only one holder notice across the whole window.
    assert err.count("held by PID 999") <= 2  # one notice + the give-up detail


def test_max_wait_zero_is_fail_fast(virtual_clock, capsys):
    """``MEMPALACE_MINE_LOCK_WAIT=0`` keeps the old fail-fast behaviour."""
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        raise MineAlreadyRunning("palace /p is held by PID 5")

    with pytest.raises(SystemExit) as exc_info:
        cli._mine_with_lock_backoff(attempt, max_wait=0.0)

    assert exc_info.value.code == 1
    assert calls["n"] == 1  # single attempt, no retry
    assert virtual_clock["sleeps"] == []
    assert "giving up" in capsys.readouterr().err


def test_success_first_try_is_silent(virtual_clock, capsys):
    """No contention → no notice, no sleeps, value returned."""
    result = cli._mine_with_lock_backoff(lambda: "ok", max_wait=30.0)
    assert result == "ok"
    assert virtual_clock["sleeps"] == []
    assert capsys.readouterr().err == ""


def test_budget_reads_env(monkeypatch):
    """_mine_lock_wait_budget honours MEMPALACE_MINE_LOCK_WAIT, with fallback."""
    monkeypatch.setenv("MEMPALACE_MINE_LOCK_WAIT", "12.5")
    assert cli._mine_lock_wait_budget() == 12.5

    monkeypatch.setenv("MEMPALACE_MINE_LOCK_WAIT", "not-a-number")
    assert cli._mine_lock_wait_budget() == 30.0

    monkeypatch.setenv("MEMPALACE_MINE_LOCK_WAIT", "-5")
    assert cli._mine_lock_wait_budget() == 30.0

    monkeypatch.delenv("MEMPALACE_MINE_LOCK_WAIT", raising=False)
    assert cli._mine_lock_wait_budget() == 30.0
