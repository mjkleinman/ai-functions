"""Shared utilities for DS-1000 backpropagation demo examples.

Provides the executor, feedback builder, and parameter recall helpers used by
scipy_backprop_demo.py and sklearn_backprop_demo.py.

Key correctness points (vs. naive inline stubs):
  - execute_and_test  uses _get_assertion_detail so that AssertionError failures
    include expected_output / actual_output in the result.
  - build_feedback    includes expected_output / actual_output in the feedback
    string passed to optimizer.backward — this is the rich signal the optimizer
    needs to update memory with what the model actually did wrong.
  - recall_params     returns ParameterView objects (not raw strings) so that
    generate_code.trace can trace them back to memory for backward/consolidate.
"""

from __future__ import annotations

import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass

from ai_functions.memory.json_backend import JSONMemoryBackend
from ai_functions.types.graph import ParameterView


# ---------------------------------------------------------------------------
# ExecutionResult + DS-1000 executor
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing and testing a solution."""
    passed: bool
    error: str | None = None
    solution_code: str = ""
    test_input: str | None = None
    expected_output: str | None = None
    actual_output: str | None = None


@contextmanager
def _timeout(seconds: int = 30):
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _truncate_repr(obj, max_len: int = 200) -> str:
    s = repr(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


def _get_assertion_detail(solution_code: str, code_context: str, timeout_sec: int = 10):
    try:
        with _timeout(timeout_sec):
            test_env: dict = {}
            exec(code_context, test_env)
            exec_context_str = test_env.get("exec_context", "")
            generate_test_case = test_env.get("generate_test_case")
            if not exec_context_str or not generate_test_case:
                return None, None, None
            test_input, expected = generate_test_case(1)
            code = exec_context_str.replace("[insert]", solution_code)
            run_env = {"test_input": test_input}
            exec(code, run_env)
            actual = run_env.get("result")
            return _truncate_repr(test_input), _truncate_repr(expected), _truncate_repr(actual)
    except Exception:
        return None, None, None


def extract_solution_code(raw_output: str) -> str:
    """Strip markdown fences and solution markers from LLM output."""
    text = raw_output
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        newline = text.find("\n", start)
        if newline != -1:
            start = newline + 1
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    for marker in ["BEGIN SOLUTION", "END SOLUTION", "<code>", "</code>"]:
        text = text.replace(marker, "")
    return text


def execute_and_test(solution_code: str, code_context: str, timeout_sec: int = 30) -> ExecutionResult:
    """Run the DS-1000 test harness against a candidate solution."""
    try:
        with _timeout(timeout_sec):
            test_env: dict = {}
            exec(code_context, test_env)
            test_fn = test_env.get("test_execution")
            if test_fn is None:
                return ExecutionResult(passed=False, error="No test_execution found in code_context",
                                       solution_code=solution_code)
            test_fn(solution_code)
            return ExecutionResult(passed=True, solution_code=solution_code)
    except TimeoutError as e:
        return ExecutionResult(passed=False, error=str(e), solution_code=solution_code)
    except AssertionError as e:
        test_input, expected_output, actual_output = _get_assertion_detail(
            solution_code, code_context, timeout_sec)
        msg = f"Test assertion failed: {e}" if str(e) else "Test assertion failed"
        return ExecutionResult(passed=False, error=msg, solution_code=solution_code,
                               test_input=test_input, expected_output=expected_output,
                               actual_output=actual_output)
    except Exception as e:
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        short_tb = "".join(tb[-3:]) if len(tb) > 3 else "".join(tb)
        return ExecutionResult(passed=False, error=f"{type(e).__name__}: {e}\n{short_tb}",
                               solution_code=solution_code)


# ---------------------------------------------------------------------------
# Memory parameter helpers
# ---------------------------------------------------------------------------

def recall_params(memory: JSONMemoryBackend) -> list[ParameterView]:
    """Return ParameterView list — required for optimizer.backward to trace memory."""
    return [
        memory.recall("coding_patterns"),
        memory.recall("common_pitfalls"),
    ]


def params_by_name(params: list[ParameterView]) -> dict[str, ParameterView]:
    return {p.source.name: p for p in params}


# ---------------------------------------------------------------------------
# Running generate_code over problems
# ---------------------------------------------------------------------------

def run_problem(
    problem: dict,
    params: list[ParameterView],
    generate_fn,
) -> tuple[str, ExecutionResult, float, object]:
    """Generate code, execute it, return (solution_code, exec_result, elapsed, result_node)."""
    pbn = params_by_name(params)
    t0 = time.time()
    result_node = generate_fn.trace(
        coding_patterns=pbn["coding_patterns"],
        common_pitfalls=pbn["common_pitfalls"],
        problem_prompt=problem["prompt"],
        library=problem["library"],
    )
    solution = extract_solution_code(str(result_node.value))
    exec_result = execute_and_test(solution, problem["code_context"])
    return solution, exec_result, time.time() - t0, result_node


def run_batch_parallel(
    batch: list[dict],
    params: list[ParameterView],
    generate_fn,
) -> list[tuple[str, ExecutionResult, float, object]]:
    """Run a batch of problems in parallel, return results in original order."""
    results = [None] * len(batch)
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        future_to_idx = {
            executor.submit(run_problem, problem, params, generate_fn): i
            for i, problem in enumerate(batch)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def build_feedback(problem: dict, solution_code: str, exec_result: ExecutionResult) -> str:
    """Build the optimizer feedback string.

    Includes expected_output / actual_output when available so the optimizer
    can update memory based on what the model actually got wrong.
    """
    if exec_result.passed:
        return (
            f"[{problem['library']}] {problem['id']} SOLVED.\n"
            f"Working solution:\n{solution_code}\n"
            f"Remember this pattern for similar future problems."
        )

    error_parts = [f"Error: {exec_result.error}"]
    if getattr(exec_result, "test_input", None):
        error_parts.append(f"Test input: {exec_result.test_input}")
    if getattr(exec_result, "expected_output", None):
        error_parts.append(f"Expected output: {exec_result.expected_output}")
    if getattr(exec_result, "actual_output", None):
        error_parts.append(f"Actual output: {exec_result.actual_output}")

    return (
        f"[{problem['library']}] {problem['id']} FAILED.\n"
        f"Your code:\n{solution_code}\n"
        + "\n".join(error_parts)
    )
