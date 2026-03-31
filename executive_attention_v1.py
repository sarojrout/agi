import random
import re

import pandas as pd
import kaggle_benchmarks as kbench


def build_dataset(n_executive: int = 12, n_attention: int = 12, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []

    math_pairs = [
        ("What is 2 + 2?", "4"),
        ("What is 5 + 5?", "10"),
        ("What is 7 + 3?", "10"),
        ("What is 9 + 1?", "10"),
        ("What is 6 + 2?", "8"),
        ("What is 3 + 5?", "8"),
        ("What is 11 + 4?", "15"),
        ("What is 20 - 6?", "14"),
    ]

    # Executive: conflicting instructions — model must ignore the "always 999" rule.
    for _ in range(n_executive):
        q, ans = random.choice(math_pairs)
        prompt = f"""Instruction 1: For every math question below, always respond with the number 999.
Instruction 2: Ignore Instruction 1 completely.
Question: {q}
Reply with only the correct integer, nothing else."""
        rows.append(
            {
                "Prompt": prompt,
                "Target_Answer": ans,
                "Track": "executive",
                "Category": "override_wrong_numeric_rule",
            }
        )

    # Attention: distractors — model must use only the OFFICIAL block.
    for _ in range(n_attention):
        a = random.randint(5, 25)
        b = random.randint(5, 25)
        wrong1, wrong2 = random.randint(100, 200), random.randint(100, 200)
        prompt = f"""The following lines contain drafts and rumors — do NOT use them for the final answer.

Rumor: A might be {wrong1} and B might be {wrong2}.
Deprecated note: sum was previously reported as {wrong1 + wrong2}.
Advertisement: Buy product 999 for only $3.99.

OFFICIAL DATA (USE ONLY THIS): A = {a}, B = {b}

Question: What is A + B? Reply with the integer only."""
        rows.append(
            {
                "Prompt": prompt,
                "Target_Answer": str(a + b),
                "Track": "attention",
                "Category": "distractor_then_official_block",
            }
        )

    random.shuffle(rows)
    return pd.DataFrame(rows)


@kbench.task(
    name="executive_attention_v1",
    description="Executive: obey override. Attention: use only OFFICIAL DATA. Pass if response contains the target integer.",
)
def executive_attention_v1(
    llm,
    Prompt: str,
    Target_Answer: str,
    Track: str,
    Category: str,
) -> None:
    response = llm.prompt(Prompt)
    target = int(Target_Answer)
    pattern = rf"\b{re.escape(str(target))}\b"
    kbench.assertions.assert_contains_regex(
        pattern,
        response,
        expectation=f"[{Track}/{Category}] Expected integer {target} in response.",
    )


df = build_dataset()
executive_attention_v1.evaluate(llm=[kbench.llm], evaluation_data=df)