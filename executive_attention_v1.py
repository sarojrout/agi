"""
Minimal Kaggle Benchmarks example — run this inside a Kaggle notebook that has
`kaggle_benchmarks` (Benchmarks kernel), not necessarily on your laptop.

What this does:
  1. Defines ONE task: answer a simple arithmetic question.
  2. Builds a small table of (Prompt, Target_Answer).
  3. Runs the same rows against several models and counts passes.

Replace model keys with whatever your notebook exposes in kbench.llms.
"""

import pandas as pd

import kaggle_benchmarks as kbench


# -----------------------------------------------------------------------------
# 1) TASK: column names in the DataFrame become function parameters after `llm`.
# -----------------------------------------------------------------------------
@kbench.task(
    name="simple_arithmetic",
    description="Model must include the correct integer in its answer.",
)
def simple_arithmetic(llm, Prompt: str, Target_Answer: str) -> None:
    response = llm.prompt(Prompt)
    target = int(Target_Answer)

    # Word boundary so "10" does not match inside "100".
    pattern = rf"\b{target}\b"
    kbench.assertions.assert_contains_regex(
        pattern,
        response,
        expectation=f"Response should contain the integer {target}.",
    )


# -----------------------------------------------------------------------------
# 2) DATA: a few addition / subtraction items (you can add more rows).
# -----------------------------------------------------------------------------
def build_rows():
    rows = [
        ("What is 17 + 25? Reply with the number only.", "42"),
        ("What is 100 - 37? Reply with the number only.", "63"),
        ("What is 8 + 9? Reply with the number only.", "17"),
        ("Compute 50 - 14. Reply with the number only.", "36"),
        ("What is 0 + 0? Reply with the number only.", "0"),
    ]
    return pd.DataFrame(
        [{"Prompt": p, "Target_Answer": a} for p, a in rows]
    )


# -----------------------------------------------------------------------------
# 3) EVALUATE: same data, several models — this is what others are doing.
# -----------------------------------------------------------------------------
def count_passed(results):
    """Best-effort pass count; shape matches sample.ipynb pattern."""
    n = 0
    for r in results:
        if getattr(r.state, "name", "") != "BENCHMARK_TASK_RUN_STATE_COMPLETED":
            continue
        if r.results and r.results[0].boolean_result:
            n += 1
    return n


def main():
    df = build_rows()

    # Pick models your environment actually lists: dir(kbench.llms) or docs.
    model_keys = [
        "google/gemini-2.5-flash",
        "google/gemma-3-12b",
    ]

    for key in model_keys:
        if key not in kbench.llms:
            print(f"Skip (not available): {key}")
            continue
        print(f"\n=== Running: {key} ===")
        results = simple_arithmetic.evaluate(
            llm=[kbench.llms[key]],
            evaluation_data=df,
        )
        passed = count_passed(results)
        print(f"Passed: {passed} / {len(df)}")


if __name__ == "__main__":
    main()
