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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

SQUAD_DATASET = "squad"
IMDB_DATASET = "imdb"
QA_TASK = "qa"
SENTIMENT_TASK = "sentiment"
DEFAULT_MODEL = os.getenv("COURSE_LLM_MODEL", "Llama-3.1-70B-Instruct-custom")


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
    print("Next step: implement prompting strategies and experiment loop.")


if __name__ == "__main__":
    main()
