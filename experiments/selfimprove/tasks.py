"""Online-loop task catalog.

The online loop needs tasks to point at *repeatedly*: each one is a tiny,
self-contained coding request with a fixture directory the harness can be told
to work inside. Tasks are generated on demand into a scratch directory so the
harness' path jail is satisfied (the harness works inside a directory, never
on the loose filesystem).

Three tasks ship here; the notebook/chart can add more. Each task is a dict
with `id`, `task` (the prompt shown to the agent), and `materialize(scratch)`
which writes the working directory and returns the path — plus a `verify`
check the online loop can use to score a done run when a *real* verifier is
wanted (the default loop uses the DeepSeek judge instead; the check is the
task's own hidden test, exactly like SWE-bench's FAIL_TO_PASS).
"""

import os
import textwrap
import tempfile


def _write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(content).lstrip())


def _materialize_broken_fixture(base, buggy, test):
    _write_file(os.path.join(base, "dice.py"), buggy)
    _write_file(os.path.join(base, "test_dice.py"), test)
    return base


def _task_fizzbuzz(scratch):
    """One classic: implement fizzbuzz with hidden tests."""
    buggy = '''\
        def fizzbuzz(n):
            if n % 15 == 0:
                return "FizzBuzz"
            if n % 3 == 0:
                return "Fizz"
            if n % 5 == 0:
                return "Buzz"
            return n            # bug: returns int, tests want str
        '''
    test = '''\
        from dice import fizzbuzz

        def test_multiple():
            assert fizzbuzz(15) == "FizzBuzz"

        def test_three():
            assert fizzbuzz(3) == "Fizz"

        def test_five():
            assert fizzbuzz(5) == "Buzz"

        def test_other_is_str():
            assert fizzbuzz(7) == "7"

        def test_all_string():
            for n in range(1, 101):
                assert isinstance(fizzbuzz(n), str)
        '''
    return _materialize_broken_fixture(scratch, buggy, test)


def _task_kata_half(scratch):
    buggy = '''\
        def halves(xs):
            mid = len(xs) // 2
            return xs[:mid], xs[:mid]    # bug: second half is wrong
        '''
    test = '''\
        from dice import halves

        def test_even():
            assert halves([1,2,3,4]) == ([1,2],[3,4])

        def test_odd():
            assert halves([10,20,30,40,50]) == ([10,20],[30,40,50])

        def test_empty():
            assert halves([]) == ([],[])
        '''
    return _materialize_broken_fixture(scratch, buggy, test)


def _task_wc(scratch):
    buggy = '''\
        import sys

        def wc(data):
            return len(data.splitlines()), len(data.split())

        if __name__ == "__main__":
            # bug: reads nothing now, so `python dice.py file` is broken
            data = ""
            lines, words = wc(data)
            print(f"{lines} {words}")
        '''
    test = '''\
        import subprocess, sys

        def test_wc():
            out = subprocess.run([sys.executable, "dice.py", "a.txt"],
                                 capture_output=True, text=True)
            assert out.stdout.strip() == "2 4"
        '''
    base = _materialize_broken_fixture(scratch, buggy, test)
    _write_file(os.path.join(base, "a.txt"), "hello world\nsecond line\n")
    return base


def _materialize_task(task, scratch_root):
    """Run a task's materializer inside its own scratch dir; returns the
    working directory the harness should operate in."""
    base = os.path.join(scratch_root, task["id"])
    task["materialize"](base)
    return base


def catalog():
    return [
        {"id": "fizzbuzz", "task": "Fix the bug in dice.py so the hidden "
                                   "tests in test_dice.py pass.",
         "materialize": _task_fizzbuzz,
         "verify": lambda d: _run_pytest(d, "test_dice.py")},
        {"id": "halves", "task": "Fix the bug in dice.py so the pairs in "
                                 "test_dice.py come out right.",
         "materialize": _task_kata_half,
         "verify": lambda d: _run_pytest(d, "test_dice.py")},
        {"id": "wc", "task": "Make dice.py a working word-count CLI: "
                             "`python dice.py a.txt` must print line and "
                             "word counts from the file argument.",
         "materialize": _task_wc,
         "verify": lambda d: _run_pytest(d, "test_dice.py")},
    ]


def materialize(task, scratch_root):
    return _materialize_task(task, scratch_root)


def _run_pytest(directory, test_file):
    """Run a task directory's test file in a subprocess. Returns True when it
    passes. This is the loop's `reward` when a real verifier is wanted; it is
    not the default — the judge is."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q"],
        cwd=directory, capture_output=True, text=True)
    return result.returncode == 0


def scaffold_root(base=None):
    """A scratch root for materialised tasks."""
    dirpath = tempfile.mkdtemp(prefix="irina_tasks_",
                               dir=base or os.environ.get("TMPDIR"))
    return dirpath