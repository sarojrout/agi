import kaggle_benchmarks as kbench
import pandas as pd
import textwrap


BUGGY_FUNCTIONS = [
    {
        "id": "sum_positive",
        "buggy": textwrap.dedent(
            """
            def sum_positive(numbers):
                total = 0
                for n in numbers:
                    if n < 0:
                        total += n  # BUG: should skip negatives
                return total
            """
        ).strip(),
        "expected_snippet": "if n > 0:",
        "forbidden_snippet": "print(",
    },
    {
        "id": "apply_discount",
        "buggy": textwrap.dedent(
            """
            def apply_discount(price, discount):
                # discount is between 0 and 1
                final_price = price + (price * discount)  # BUG: should subtract
                return final_price
            """
        ).strip(),
        "expected_snippet": "price - (price * discount)",
        "forbidden_snippet": "input(",
    },
]


def build_dev_dataset() -> pd.DataFrame:
    rows = []
    for item in BUGGY_FUNCTIONS:
        prompt = f"""
You are a senior Python developer.

Here is a BUGGY function:

```python
{item["buggy"]}
```

Fix ONLY the bug in the logic.
DO NOT change the function name or parameters.
DO NOT add logging, print statements, or comments.
Return ONLY the corrected function as valid Python code (no explanation, no markdown).
"""
        rows.append(
            {
                "Prompt": textwrap.dedent(prompt).strip(),
                "FnId": item["id"],
                "ExpectedSnippet": item["expected_snippet"],
                "ForbiddenSnippet": item["forbidden_snippet"],
            }
        )
    return pd.DataFrame(rows)


@kbench.task(
    name="developer_bugfix_v1",
    description="Given a small buggy Python function, return ONLY the fixed function.",
)
def developer_bugfix_v1(
    llm,
    Prompt: str,
    FnId: str,
    ExpectedSnippet: str,
    ForbiddenSnippet: str,
) -> None:
    response = llm.prompt(Prompt)

    stripped = response.lstrip()
    kbench.assertions.assert_contains_regex(
        r"^def\s+\w+\(",
        stripped,
        expectation=f"[{FnId}] Response should start with a Python function definition.",
    )

    kbench.assertions.assert_in(
        ExpectedSnippet,
        response,
        expectation=f"[{FnId}] Response should include the fixed logic snippet: {ExpectedSnippet!r}.",
    )

    if ForbiddenSnippet:
        kbench.assertions.assert_not_in(
            ForbiddenSnippet,
            response,
            expectation=f"[{FnId}] Response should NOT contain: {ForbiddenSnippet!r}.",
        )

    kbench.assertions.assert_not_in(
        "```",
        response,
        expectation=f"[{FnId}] Response must be raw Python, not wrapped in markdown fences.",
    )


df = build_dev_dataset()
developer_bugfix_v1.evaluate(llm=[kbench.llm], evaluation_data=df)

