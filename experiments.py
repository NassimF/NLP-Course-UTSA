#!/usr/bin/env python3
"""
experiments.py - CS6263 Assignment 1 (Part 2: Prompt Engineering Experiments)

Initial scaffold:
- CLI and config parsing
- Shared constants for datasets/tasks
- Entry point for step-by-step implementation
"""

import argparse
import csv
import json
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
    use_local_smoke_data: bool


@dataclass
class ExperimentRecord:
    task: str
    technique: str
    example_id: str
    dataset_name: str
    split: str
    input_payload: Dict[str, Any]
    gold: Dict[str, Any]
    user_prompt: str
    system_prompt: str
    prompt: str
    raw_response: str
    parsed_final_answer: str
    parse_failed: bool
    format_violation: bool
    latency_ms: float
    error: str
    gold_answers: Optional[List[str]] = None
    gold_label: Optional[int] = None


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def normalize_qa_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = _remove_articles(no_punc)
    return " ".join(no_articles.split())


def normalize_sentiment_label(text: str) -> str:
    if not text:
        return "unknown"
    value = text.strip().lower()
    if "positive" in value or value in {"pos", "1"}:
        return "positive"
    if "negative" in value or value in {"neg", "0"}:
        return "negative"
    return "unknown"


def _gold_sentiment_label(label: Optional[int]) -> str:
    if label == 1:
        return "positive"
    if label == 0:
        return "negative"
    return "unknown"


def qa_exact_match(prediction: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    pred_norm = normalize_qa_text(prediction)
    for gold in gold_answers:
        if pred_norm == normalize_qa_text(gold):
            return 1.0
    return 0.0


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_qa_text(prediction).split()
    gold_tokens = normalize_qa_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_token_f1(prediction: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(_token_f1(prediction, gold) for gold in gold_answers)


def sentiment_accuracy(records: List[ExperimentRecord]) -> float:
    valid = [r for r in records if r.task == SENTIMENT_TASK and r.gold_label is not None]
    if not valid:
        return 0.0
    correct = 0
    for record in valid:
        pred = normalize_sentiment_label(record.parsed_final_answer)
        gold = _gold_sentiment_label(record.gold_label)
        if pred == gold:
            correct += 1
    return correct / len(valid)


def sentiment_macro_f1(records: List[ExperimentRecord]) -> float:
    valid = [r for r in records if r.task == SENTIMENT_TASK and r.gold_label is not None]
    if not valid:
        return 0.0
    labels = ("positive", "negative")
    f1_scores: List[float] = []
    for label in labels:
        tp = fp = fn = 0
        for record in valid:
            pred = normalize_sentiment_label(record.parsed_final_answer)
            gold = _gold_sentiment_label(record.gold_label)
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores)


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


def _strip_wrapping_pairs(text: str) -> str:
    value = text.strip()
    wrappers = (("<", ">"), ("[", "]"), ("(", ")"), ('"', '"'), ("'", "'"), ("`", "`"))
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for left, right in wrappers:
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : -len(right)].strip()
                changed = True
    return value


def _extract_sentiment_label(text: str) -> Tuple[str, bool]:
    cleaned = _strip_wrapping_pairs(text)
    labels = re.findall(r"(?i)\b(positive|negative)\b", cleaned)
    if not labels:
        return "", False
    unique = {label.lower() for label in labels}
    if len(unique) == 1:
        return labels[-1].lower(), True
    return "", False


def _final_answer_candidate(raw_text: str, strict: bool) -> Optional[str]:
    if strict:
        matches = re.findall(r"(?im)^Final Answer:\s*(.+?)\s*$", raw_text.strip())
        return matches[-1].strip() if matches else None

    matches = re.findall(r"(?is)Final Answer:\s*(.+)", raw_text)
    if not matches:
        return None
    tail = matches[-1].strip()
    for line in tail.splitlines():
        if line.strip():
            return line.strip()
    return tail


def extract_final_answer(raw_text: str, task: Optional[str] = None) -> Tuple[str, bool, bool]:
    # Parse "Final Answer:" in two passes:
    # 1) strict line-start match, 2) relaxed fallback anywhere in the response.
    # format_violation flags cases where the fallback path was required.
    if not raw_text:
        return "", True, False

    strict_candidate = _final_answer_candidate(raw_text, strict=True)
    if strict_candidate is not None:
        if task == SENTIMENT_TASK:
            label, ok = _extract_sentiment_label(strict_candidate)
            if ok:
                return label, False, False
            return _strip_wrapping_pairs(strict_candidate), True, False
        return _strip_wrapping_pairs(strict_candidate), False, False

    relaxed_candidate = _final_answer_candidate(raw_text, strict=False)
    if relaxed_candidate is not None:
        if task == SENTIMENT_TASK:
            label, ok = _extract_sentiment_label(relaxed_candidate)
            if ok:
                return label, False, True
            return _strip_wrapping_pairs(relaxed_candidate), True, True
        return _strip_wrapping_pairs(relaxed_candidate), False, True

    if task == SENTIMENT_TASK:
        label, ok = _extract_sentiment_label(raw_text)
        if ok:
            return label, False, True
    return raw_text.strip(), True, True


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


def _record_system_prompt(technique: str) -> str:
    override = _system_prompt_for_technique(technique)
    if override is not None:
        return override
    return "DEFAULT (api_basics.SYSTEM_PROMPT)"


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run Part 2 prompt-engineering experiments (QA + sentiment)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--use-local-smoke-data",
        action="store_true",
        help="Use tiny built-in QA/sentiment examples instead of downloading datasets.",
    )
    args = parser.parse_args()

    return ExperimentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        output_dir=args.output_dir,
        use_local_smoke_data=args.use_local_smoke_data,
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


def _local_qa_examples() -> List[QAExample]:
    return [
        QAExample(
            example_id="local-qa-1",
            question="What is the capital of France?",
            context="France's capital city is Paris, known for the Eiffel Tower.",
            gold_answers=["Paris"],
        ),
        QAExample(
            example_id="local-qa-2",
            question="Which planet is largest in the Solar System?",
            context="Jupiter is the largest planet in the Solar System.",
            gold_answers=["Jupiter"],
        ),
    ]


def _local_sentiment_examples() -> List[SentimentExample]:
    return [
        SentimentExample(
            example_id="local-imdb-1",
            review_text="I loved this movie. Great acting and story.",
            gold_label=1,
        ),
        SentimentExample(
            example_id="local-imdb-2",
            review_text="This film was boring and way too long.",
            gold_label=0,
        ),
    ]


def load_qa_examples(n_samples: int, seed: int, use_local_smoke_data: bool = False) -> List[QAExample]:
    if use_local_smoke_data:
        local = _local_qa_examples()
        return local[: min(n_samples, len(local))]

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


def load_sentiment_examples(
    n_samples: int, seed: int, use_local_smoke_data: bool = False
) -> List[SentimentExample]:
    if use_local_smoke_data:
        local = _local_sentiment_examples()
        return local[: min(n_samples, len(local))]

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
    dataset_name = "local_smoke_squad" if config.use_local_smoke_data else SQUAD_DATASET
    split_name = "local" if config.use_local_smoke_data else "validation"
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
            parsed, parse_failed, format_violation = extract_final_answer(
                raw_response, task=QA_TASK
            )
            system_prompt = _record_system_prompt(technique)
            # Store both raw text and parsed final answer for reproducibility/evaluation.
            records.append(
                ExperimentRecord(
                    task=QA_TASK,
                    technique=technique,
                    example_id=example.example_id,
                    dataset_name=dataset_name,
                    split=split_name,
                    input_payload={
                        "question": example.question,
                        "context": example.context,
                    },
                    gold={"answers": example.gold_answers},
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_final_answer=parsed,
                    parse_failed=parse_failed,
                    format_violation=format_violation,
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
    dataset_name = "local_smoke_imdb" if config.use_local_smoke_data else IMDB_DATASET
    split_name = "local" if config.use_local_smoke_data else "test"
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
            parsed, parse_failed, format_violation = extract_final_answer(
                raw_response, task=SENTIMENT_TASK
            )
            system_prompt = _record_system_prompt(technique)
            # Save parsed label and gold label for later scoring (accuracy/F1).
            records.append(
                ExperimentRecord(
                    task=SENTIMENT_TASK,
                    technique=technique,
                    example_id=example.example_id,
                    dataset_name=dataset_name,
                    split=split_name,
                    input_payload={"review_text": example.review_text},
                    gold={"label": example.gold_label},
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_final_answer=parsed,
                    parse_failed=parse_failed,
                    format_violation=format_violation,
                    latency_ms=latency_ms,
                    error=error,
                    gold_answers=None,
                    gold_label=example.gold_label,
                )
            )
    return records


def summarize_metrics(records: List[ExperimentRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.task, record.technique)].append(record)

    rows: List[Dict[str, Any]] = []
    for (task, technique), group in sorted(grouped.items()):
        base_row: Dict[str, Any] = {
            "task": task,
            "technique": technique,
            "num_samples": len(group),
            "error_count": sum(1 for r in group if r.error),
            "parse_fail_count": sum(1 for r in group if r.parse_failed),
            "format_violation_count": sum(1 for r in group if r.format_violation),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in group) / len(group) if group else 0.0, 3
            ),
        }

        if task == QA_TASK:
            em_scores = [qa_exact_match(r.parsed_final_answer, r.gold_answers or []) for r in group]
            f1_scores = [qa_token_f1(r.parsed_final_answer, r.gold_answers or []) for r in group]
            base_row["exact_match"] = round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0
            base_row["token_f1"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
            base_row["accuracy"] = ""
            base_row["macro_f1"] = ""
        else:
            base_row["exact_match"] = ""
            base_row["token_f1"] = ""
            base_row["accuracy"] = round(sentiment_accuracy(group), 4)
            base_row["macro_f1"] = round(sentiment_macro_f1(group), 4)

        rows.append(base_row)
    return rows


def write_experiment_runs_jsonl(path: Path, records: List[ExperimentRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = asdict(record)
            if record.task == QA_TASK:
                row["qa_exact_match"] = qa_exact_match(record.parsed_final_answer, record.gold_answers or [])
                row["qa_token_f1"] = qa_token_f1(record.parsed_final_answer, record.gold_answers or [])
                row["normalized_prediction"] = normalize_qa_text(record.parsed_final_answer)
            else:
                row["normalized_prediction"] = normalize_sentiment_label(record.parsed_final_answer)
                row["gold_label_name"] = _gold_sentiment_label(record.gold_label)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary_csv(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "technique",
        "num_samples",
        "error_count",
        "parse_fail_count",
        "format_violation_count",
        "avg_latency_ms",
        "exact_match",
        "token_f1",
        "accuracy",
        "macro_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


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
    print(f"Use local smoke data: {config.use_local_smoke_data}")

    qa_examples = load_qa_examples(
        config.samples_per_task, config.seed, config.use_local_smoke_data
    )
    sentiment_examples = load_sentiment_examples(
        config.samples_per_task, config.seed, config.use_local_smoke_data
    )
    print(f"Loaded QA examples: {len(qa_examples)}")
    print(f"Loaded sentiment examples: {len(sentiment_examples)}")

    qa_records = run_qa_experiments(qa_examples, config)
    sentiment_records = run_sentiment_experiments(sentiment_examples, config)
    all_records = qa_records + sentiment_records
    error_count = sum(1 for r in all_records if r.error)
    print(f"Collected records: {len(all_records)}")
    print(f"Records with errors: {error_count}")

    summary_rows = summarize_metrics(all_records)
    runs_path = config.output_dir / "experiment_runs.jsonl"
    summary_path = config.output_dir / "summary_metrics.csv"
    write_experiment_runs_jsonl(runs_path, all_records)
    write_summary_csv(summary_path, summary_rows)
    print(f"Wrote runs JSONL: {runs_path}")
    print(f"Wrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
