#!/usr/bin/env python3
"""
experiments.py - CS6263 Assignment 1 (Part 2: Prompt Engineering Experiments)

Initial scaffold:
- CLI and config parsing
- Shared constants for datasets/tasks
- Entry point for step-by-step implementation
"""

import argparse
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SQUAD_DATASET = "squad"
IMDB_DATASET = "imdb"
QA_TASK = "qa"
SENTIMENT_TASK = "sentiment"
DEFAULT_MODEL = os.getenv("COURSE_LLM_MODEL", "Llama-3.1-70B-Instruct-custom")
TECHNIQUE_ZERO_SHOT = "zero_shot"
TECHNIQUE_FEW_SHOT = "few_shot"
TECHNIQUE_COT = "chain_of_thought"
TECHNIQUE_CUSTOM = "custom_variation"
PROMPT_TECHNIQUES = (
    TECHNIQUE_ZERO_SHOT,
    TECHNIQUE_FEW_SHOT,
    TECHNIQUE_COT,
    TECHNIQUE_CUSTOM,
)


@dataclass #decorator to generate common class methods likt __init__ automatically
class QAExample:
    example_id: str
    question: str
    context: str
    gold_answers: List[str]


@dataclass
class SentimentExample:
    example_id: str
    review_text: str
    gold_label: int


@dataclass
class ExperimentConfig:
    model: str
    temperature: float
    max_tokens: int
    samples_per_task: int
    seed: int
    output_dir: Path


@dataclass
class ExperimentRecord:
    task: str
    technique: str
    example_id: str
    prompt: str
    raw_response: str
    parsed_final_answer: str
    latency_ms: float
    error: str
    gold_answers: Optional[List[str]] = None
    gold_label: Optional[int] = None


def _validate_technique(technique: str) -> None:
    if technique not in PROMPT_TECHNIQUES:
        raise ValueError(f"Unsupported technique: {technique}")


def build_qa_prompt(example: QAExample, technique: str) -> str:
    _validate_technique(technique)
    shared = (
        "Task: Answer the question using only the context.\n"
        "If the answer is not present, output exactly: Final Answer: unanswerable\n"
        "Always end with one line formatted exactly as:\n"
        "Final Answer: <short answer>\n"
    )

    if technique == TECHNIQUE_ZERO_SHOT:
        strategy = "Answer directly and concisely."
    elif technique == TECHNIQUE_FEW_SHOT:
        strategy = (
            "Use the following examples as guidance.\n"
            "Example 1:\n"
            "Context: Paris is the capital city of France.\n"
            "Question: What is the capital of France?\n"
            "Final Answer: Paris\n\n"
            "Example 2:\n"
            "Context: The largest planet in the Solar System is Jupiter.\n"
            "Question: Which planet is the largest in the Solar System?\n"
            "Final Answer: Jupiter\n"
        )
    elif technique == TECHNIQUE_COT:
        strategy = (
            "Reason step by step before deciding.\n"
            "Then provide exactly one final line in the required format."
        )
    else:
        strategy = (
            "First identify a short evidence phrase from context.\n"
            "Then provide the final answer.\n"
            "Format:\n"
            "Evidence: <quote>\n"
            "Final Answer: <short answer>\n"
        )

    return (
        f"{shared}\n"
        f"Strategy:\n{strategy}\n\n"
        f"Context:\n{example.context}\n\n"
        f"Question:\n{example.question}\n"
    )


def build_sentiment_prompt(example: SentimentExample, technique: str) -> str:
    _validate_technique(technique)
    shared = (
        "Task: Classify the review sentiment.\n"
        "Allowed labels: positive, negative\n"
        "Always end with one line formatted exactly as:\n"
        "Final Answer: <positive|negative>\n"
    )

    if technique == TECHNIQUE_ZERO_SHOT:
        strategy = "Classify sentiment directly from the review."
    elif technique == TECHNIQUE_FEW_SHOT:
        strategy = (
            "Use the following examples as guidance.\n"
            "Example 1:\n"
            "Review: I loved this movie and would watch it again.\n"
            "Final Answer: positive\n\n"
            "Example 2:\n"
            "Review: The plot was boring and I regret watching it.\n"
            "Final Answer: negative\n"
        )
    elif technique == TECHNIQUE_COT:
        strategy = (
            "Reason briefly about sentiment cues before deciding.\n"
            "Then provide exactly one final line in the required format."
        )
    else:
        strategy = (
            "Use a structured approach:\n"
            "Positive cues: <short list>\n"
            "Negative cues: <short list>\n"
            "Final Answer: <positive|negative>\n"
        )

    return (
        f"{shared}\n"
        f"Strategy:\n{strategy}\n\n"
        f"Review:\n{example.review_text}\n"
    )


def extract_final_answer(raw_text: str) -> str:
    # Parse the canonical answer line used by all prompting strategies.
    # If the expected line is missing, fall back to full trimmed output.
    if not raw_text:
        return ""
    matches = re.findall(r"(?im)^Final Answer:\s*(.+?)\s*$", raw_text.strip())
    if matches:
        return matches[-1].strip()
    return raw_text.strip()


def _system_prompt_for_technique(technique: str) -> Optional[str]:
    # Only CoT runs override the Part-1 default behavior to allow reasoning text.
    # Other techniques use api_basics.py defaults for consistency.
    if technique == TECHNIQUE_COT:
        return (
            "You are a helpful assistant. Reason step-by-step when useful. "
            "Do not use <think> tags. Follow formatting instructions exactly."
        )
    return None


def call_llm(prompt: str, config: ExperimentConfig, technique: str) -> str:
    # Lazy import keeps setup commands (e.g., --help) usable without API deps/env.
    # This function is the single call path to Part-1 query_llm for all experiments.
    from api_basics import query_llm

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    system_prompt = _system_prompt_for_technique(technique)
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return query_llm(prompt, **kwargs)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run Part 2 prompt-engineering experiments (QA + sentiment)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    return ExperimentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def _sample_hf_split(
    dataset_name: str, split_name: str, n_samples: int, seed: int
) -> Tuple[object, List[int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install `datasets` first (pip install datasets)."
        ) from exc

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    split = load_dataset(dataset_name, split=split_name)
    split_size = len(split)
    if n_samples > split_size:
        raise ValueError(
            f"Requested {n_samples} samples from {dataset_name}/{split_name}, "
            f"but split has only {split_size} rows."
        )

    sampled_indices = random.Random(seed).sample(range(split_size), n_samples)
    sampled = split.select(sampled_indices)
    return sampled, sampled_indices


def load_qa_examples(n_samples: int, seed: int) -> List[QAExample]:
    sampled, _ = _sample_hf_split(SQUAD_DATASET, "validation", n_samples, seed)
    rows: List[QAExample] = []
    for row in sampled:
        answers = row.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        rows.append(
            QAExample(
                example_id=str(row["id"]),
                question=row["question"],
                context=row["context"],
                gold_answers=[str(x) for x in texts],
            )
        )
    return rows


def load_sentiment_examples(n_samples: int, seed: int) -> List[SentimentExample]:
    sampled, sampled_indices = _sample_hf_split(IMDB_DATASET, "test", n_samples, seed)
    rows: List[SentimentExample] = []
    for sampled_idx, row in enumerate(sampled):
        source_index = sampled_indices[sampled_idx]
        rows.append(
            SentimentExample(
                example_id=f"imdb-test-{source_index}",
                review_text=row["text"],
                gold_label=int(row["label"]),
            )
        )
    return rows


def run_qa_experiments(
    examples: List[QAExample], config: ExperimentConfig
) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []
    # Evaluate each prompting strategy on the same QA sample set.
    for technique in PROMPT_TECHNIQUES:
        for example in examples:
            # Build a technique-specific prompt from the same underlying example.
            prompt = build_qa_prompt(example, technique)
            # Measure end-to-end latency of the LLM call for cost/latency analysis.
            start = time.perf_counter()
            raw_response = ""
            error = ""
            try:
                raw_response = call_llm(prompt, config, technique)
            except Exception as exc:
                # Keep running if one call fails; capture failure for later analysis.
                error = str(exc)
            latency_ms = (time.perf_counter() - start) * 1000.0
            parsed = extract_final_answer(raw_response)
            # Store both raw text and parsed final answer for reproducibility/evaluation.
            records.append(
                ExperimentRecord(
                    task=QA_TASK,
                    technique=technique,
                    example_id=example.example_id,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_final_answer=parsed,
                    latency_ms=latency_ms,
                    error=error,
                    gold_answers=example.gold_answers,
                    gold_label=None,
                )
            )
    return records


def run_sentiment_experiments(
    examples: List[SentimentExample], config: ExperimentConfig
) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []
    # Same evaluation shape as QA so comparisons are controlled across tasks.
    for technique in PROMPT_TECHNIQUES:
        for example in examples:
            # Build prompt with sentiment-specific instructions for this technique.
            prompt = build_sentiment_prompt(example, technique)
            # Track latency per call to compare efficiency across prompt styles.
            start = time.perf_counter()
            raw_response = ""
            error = ""
            try:
                raw_response = call_llm(prompt, config, technique)
            except Exception as exc:
                # Record the error and continue to avoid aborting the whole run.
                error = str(exc)
            latency_ms = (time.perf_counter() - start) * 1000.0
            parsed = extract_final_answer(raw_response)
            # Save parsed label and gold label for later scoring (accuracy/F1).
            records.append(
                ExperimentRecord(
                    task=SENTIMENT_TASK,
                    technique=technique,
                    example_id=example.example_id,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_final_answer=parsed,
                    latency_ms=latency_ms,
                    error=error,
                    gold_answers=None,
                    gold_label=example.gold_label,
                )
            )
    return records


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("== Part 2 Experiments ==")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Samples per task: {config.samples_per_task}")
    print(f"Seed: {config.seed}")
    print(f"Output dir: {config.output_dir}")

    qa_examples = load_qa_examples(config.samples_per_task, config.seed)
    sentiment_examples = load_sentiment_examples(config.samples_per_task, config.seed)
    print(f"Loaded QA examples: {len(qa_examples)}")
    print(f"Loaded sentiment examples: {len(sentiment_examples)}")

    qa_records = run_qa_experiments(qa_examples, config)
    sentiment_records = run_sentiment_experiments(sentiment_examples, config)
    all_records = qa_records + sentiment_records
    error_count = sum(1 for r in all_records if r.error)
    print(f"Collected records: {len(all_records)}")
    print(f"Records with errors: {error_count}")
    print("Next step: add JSONL/CSV export and scoring metrics.")


if __name__ == "__main__":
    main()
