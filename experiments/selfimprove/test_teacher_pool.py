"""Offline checks for the teacher's parse helpers and the insight pool.

Run:  python -m experiments.selfimprove.test_teacher_pool
No torch, no network, no DeepSeek key required — the parse functions take the
reply strings a judge actually produces, and the pool persists to a temp file,
so the offline-learning data plumbing is proven before any Colab runtime.
"""

import os
import tempfile

from . import insight_pool, teacher


def test_parse_grade(_tmp=None):
    assert teacher.parse_grade("SCORE 7 solid exploration") == 7.0
    assert teacher.parse_grade("Score: 8.5 good but verbose") == 8.5
    assert teacher.parse_grade("the score is 10\nflawless") == 10.0
    assert teacher.parse_grade("no verdict") is None
    assert teacher.parse_grade("") is None


def test_parse_golden(_tmp=None):
    trace = "1. read README.md\n2. find the bug in main.py"
    assert teacher.parse_golden(trace) == trace
    fenced = f"```text\n{trace}\n```"
    assert teacher.parse_golden(fenced) == trace
    assert teacher.parse_golden("   ") is None
    assert teacher.parse_golden("") is None


def _transcript_ok(score=7.0):
    return (score,
            [{"role": "user", "text": "find and fix the bug in main.py"}])


class StubTeacher:
    def __init__(self, score, golden=None, fail_extract=False):
        self.score = score
        self.golden = golden or {"issue": "q", "golden": "g"}
        self.fail_extract = fail_extract

    def grade(self, messages):
        return self.score

    def extract_golden(self, issue, messages):
        return None if self.fail_extract else self.golden


def test_pool_roundtrip(tmp):
    pool = insight_pool.InsightPool(
        os.path.join(tmp, "insights.json"), teacher=StubTeacher(7.0))
    msg = pool.bank("fix the bug", [{"role": "user", "text": "x"}])
    assert "banked" in msg
    assert pool.status()["count"] == 1
    # reload from disk — persisted
    pool2 = insight_pool.InsightPool(
        os.path.join(tmp, "insights.json"), teacher=StubTeacher(7.0))
    assert pool2.status()["count"] == 1
    drained = pool2.drain()
    assert len(drained) == 1
    assert pool2.status()["count"] == 0


def test_pool_rejects_bad_runs(tmp):
    pool = insight_pool.InsightPool(
        os.path.join(tmp, "insights.json"), teacher=StubTeacher(2.0))
    msg = pool.bank("fix the bug", [{"role": "user", "text": "x"}])
    assert "not banked" in msg
    assert pool.status()["count"] == 0


def test_pool_ready_threshold(tmp):
    pool = insight_pool.InsightPool(
        os.path.join(tmp, "insights.json"), threshold=2,
        teacher=StubTeacher(7.0))
    assert pool.status()["ready"] is False
    pool.bank("q1", [{"role": "user", "text": "a"}])
    pool.bank("q2", [{"role": "user", "text": "b"}])
    assert pool.status()["ready"] is True


def run(tmp):
    tests = [test_parse_grade, test_parse_golden, test_pool_roundtrip,
             test_pool_rejects_bad_runs, test_pool_ready_threshold]
    failures = 0
    for fn in tests:
        try:
            fn(tmp)
            print(f"ok   {fn.__name__}")
        except Exception:
            import traceback
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"{len(tests)} tests, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        raise SystemExit(run(tmp))