# -*- coding: utf-8 -*-
"""
run_testbed.py

Entry point for Piastra's testbed: runs the sanity / conservation /
convergence / robustness suites under tests/ and prints a single pass/fail
report.

Usage
-----
    python run_testbed.py                      # everything
    python run_testbed.py --suite sanity        # one suite
    python run_testbed.py --suite sanity,conservation
    python run_testbed.py -q                    # only print failures

Each suite is a plain module of ``test_*`` functions (no pytest dependency
required -- this script discovers and runs them itself with `inspect`), so
the same files also work with ``pytest tests/`` for anyone who prefers that
runner.

Exit code is 0 if every test passed, 1 otherwise (suitable for CI).

Author: mrkondratyev
"""

import argparse
import contextlib
import inspect
import io
import sys
import time
import traceback

import tests.test_sanity as test_sanity
import tests.test_conservation as test_conservation
import tests.test_convergence as test_convergence
import tests.test_robustness as test_robustness

SUITES = {
    "sanity":       test_sanity,
    "conservation": test_conservation,
    "convergence":  test_convergence,
    "robustness":   test_robustness,
}


def _discover(module):
    """Return the module's test_* functions, in source order."""
    funcs = [obj for name, obj in vars(module).items()
             if name.startswith("test_") and inspect.isfunction(obj)]
    funcs.sort(key=lambda f: inspect.getsourcelines(f)[1])
    return funcs


def run_suite(name, module, quiet=False):
    """
    Run every test_* function in `module`, printing one line each.

    Each test's own stdout (every IC function prints a one-line
    description; the testbed runs dozens to hundreds of them) is captured
    and only shown when that test fails or errors -- the usual test-runner
    convention, and the only way the per-test report stays readable.

    Returns
    -------
    results : list of (name, status, message, elapsed)
        status is one of 'PASS', 'FAIL', 'ERROR'.
    """
    print(f"\n{'=' * 70}\n  SUITE: {name}\n{'=' * 70}")
    results = []
    for fn in _discover(module):
        t0 = time.time()
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                fn()
            status, msg = "PASS", ""
        except AssertionError as e:
            status, msg = "FAIL", str(e)
        except Exception as e:
            status, msg = "ERROR", f"{type(e).__name__}: {e}"
            msg += "\n" + traceback.format_exc()
        elapsed = time.time() - t0
        results.append((fn.__name__, status, msg, elapsed))

        if status != "PASS" or not quiet:
            print(f"  [{status:5s}] {fn.__name__:48s} ({elapsed:6.2f}s)")
        if status != "PASS":
            stdout = captured.getvalue()
            if stdout:
                print("           --- captured stdout ---")
                for line in stdout.splitlines():
                    print(f"           {line}")
            if msg:
                print("           --- failure ---")
                for line in msg.splitlines():
                    print(f"           {line}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the Piastra testbed.")
    parser.add_argument("--suite", default="sanity,conservation,convergence,robustness",
                         help="comma-separated subset of: " + ",".join(SUITES))
    parser.add_argument("-q", "--quiet", action="store_true",
                         help="only print failing/erroring tests")
    args = parser.parse_args()

    chosen = [s.strip() for s in args.suite.split(",") if s.strip()]
    for s in chosen:
        if s not in SUITES:
            parser.error(f"unknown suite '{s}'. Expected one of: {', '.join(SUITES)}")

    t_start = time.time()
    all_results = {}
    for name in chosen:
        all_results[name] = run_suite(name, SUITES[name], quiet=args.quiet)
    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    n_pass = n_fail = n_error = 0
    for name, results in all_results.items():
        p = sum(1 for r in results if r[1] == "PASS")
        f = sum(1 for r in results if r[1] == "FAIL")
        e = sum(1 for r in results if r[1] == "ERROR")
        n_pass += p; n_fail += f; n_error += e
        print(f"  {name:14s} {p:4d} passed, {f:4d} failed, {e:4d} errored "
              f"(of {len(results)})")

    n_total = n_pass + n_fail + n_error
    print(f"\n  TOTAL: {n_pass}/{n_total} passed "
          f"({n_fail} failed, {n_error} errored) in {total_elapsed:.1f}s")
    print("=" * 70)

    ok = (n_fail == 0 and n_error == 0)
    print("RESULT: " + ("OK" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
