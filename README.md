My NLP course at UTSA

Instructor: Dr. Rad

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai datasets
```

Set your API key and run:

```bash
export COURSE_LLM_API_KEY="<your_key>"
python api_basics.py --temperature 0.7 --max-tokens 200
python experiments.py --samples-per-task 20 --output-dir outputs/final_hf_20
```

If running on the DGX server, avoid connection issues with:

```bash
export NO_PROXY="10.246.100.230,localhost,127.0.0.1"
```

## plan

- Build `experiments.py` for Assignment Part 2 using the existing `query_llm()` wrapper from `api_basics.py`.
- Run experiments on two tasks:
  - Question Answering using Hugging Face `squad` (validation split)
  - Sentiment Analysis using Hugging Face `imdb` (test split)
- Compare 4 prompting strategies for each task:
  - zero-shot
  - few-shot (2-3 examples)
  - chain-of-thought
  - custom variation
- Keep model settings constant across all comparisons (`temperature`, `max_tokens`, model).
- Use 20 sampled examples per task (seeded for reproducibility).
- Enforce parseable outputs with a required `Final Answer:` line.
- Track response latency per example and log full prompts used for each strategy.
- Evaluate outputs:
  - QA: Exact Match (EM) + token-level F1
  - Sentiment: accuracy (and optional macro-F1)
- Save deliverables:
  - raw per-example outputs to JSONL
  - aggregated metrics to CSV
- Update `query_llm()` with an optional `system_prompt` override so CoT experiments can run without changing Part 1 defaults.

### Custom Strategy: Structured Output Format

- The custom prompting strategy uses a fixed output schema instead of free-form text.
- QA prompt format asks for:
  - `Evidence: <quote from context>`
  - `Final Answer: <short answer>`
- Sentiment prompt format asks for:
  - `Positive cues: <short list>`
  - `Negative cues: <short list>`
  - `Final Answer: positive` or `Final Answer: negative`
- This helps by making responses easier to parse, more consistent across runs, and less ambiguous during scoring.

### Recent Implementation Update (Execution Skeleton)

- `api_basics.py -> query_llm(..., system_prompt=...)`: added optional `system_prompt` override so experiments can change system instructions per technique (default behavior remains unchanged).
- `experiments.py -> TECHNIQUE_* constants + PROMPT_TECHNIQUES`: defines the four required prompting strategies in one place.
- `experiments.py -> ExperimentRecord`: stores one result row per model call (task, technique, prompt, raw output, parsed answer, latency, error, gold data).
- `experiments.py -> _validate_technique(technique)`: guards against unsupported strategy names.
- `experiments.py -> build_qa_prompt(...)`: builds QA prompts for zero-shot, few-shot, chain-of-thought, and structured-output custom strategy.
- `experiments.py -> build_sentiment_prompt(...)`: builds sentiment prompts for the same four strategies.
- `experiments.py -> extract_final_answer(raw_text, task=...)`: extracts the `Final Answer:` value with task-aware parsing for consistent downstream scoring.
- `experiments.py -> _system_prompt_for_technique(technique)`: applies a CoT-specific system prompt override and leaves other techniques on defaults.
- `experiments.py -> call_llm(...)`: central LLM call path for experiments using lazy import of Part 1 `query_llm`.
- `experiments.py -> run_qa_experiments(...)`: executes all techniques across QA samples, captures latency/errors, and records outputs.
- `experiments.py -> run_sentiment_experiments(...)`: executes all techniques across sentiment samples with the same controlled structure.
- `experiments.py -> main()`: now runs dataset loading + execution skeleton and prints total records/error count.

### Recent Implementation Update (Metrics + Export)

- `experiments.py -> normalize_qa_text(text)`: normalizes QA text (lowercase, punctuation/article removal, whitespace cleanup) before EM/F1 scoring.
- `experiments.py -> normalize_sentiment_label(text)`: maps model sentiment output to `positive`, `negative`, or `unknown`.
- `experiments.py -> qa_exact_match(prediction, gold_answers)`: computes exact match against SQuAD references after normalization.
- `experiments.py -> qa_token_f1(prediction, gold_answers)`: computes token-level F1 and uses best score across multiple reference answers.
- `experiments.py -> sentiment_accuracy(records)`: computes sentiment classification accuracy from parsed outputs.
- `experiments.py -> sentiment_macro_f1(records)`: computes macro-F1 over positive/negative classes.
- `experiments.py -> summarize_metrics(records)`: aggregates metrics by `(task, technique)` including latency and error counts.
- `experiments.py -> write_experiment_runs_jsonl(path, records)`: exports per-example raw/parsed outputs and normalization/scoring fields.
- `experiments.py -> write_summary_csv(path, summary_rows)`: exports task/technique aggregate metrics to CSV.
- `experiments.py -> main()`: now writes `outputs/experiment_runs.jsonl` and `outputs/summary_metrics.csv` after running experiments.

### Recent Implementation Update (Local Smoke Validation)

- Added local smoke-data mode in `experiments.py` via `--use-local-smoke-data` to run validation without downloading Hugging Face datasets.
- Added built-in tiny QA/sentiment samples used only for smoke validation.
- Extended loaders to switch between Hugging Face data (default) and local smoke data.
- Added explicit schema fields to each record (`dataset_name`, `split`, `input_payload`, `gold`, `user_prompt`, `system_prompt`) plus `parse_failed`.
- Added `parse_fail_count` to summary aggregation and CSV export.
- Verified end-to-end smoke run:
  - `outputs/smoke_validation_local/experiment_runs.jsonl`
  - `outputs/smoke_validation_local/summary_metrics.csv`
  - `Records with errors: 0`

### Recent Implementation Update (Robust Final-Answer Parsing)

- `experiments.py -> extract_final_answer(...)` now uses a two-pass parser:
  - strict parse: expects `Final Answer:` at line start (the required format)
  - relaxed parse: if strict fails, recovers `Final Answer:` even when it appears mid-line
- Added helper functions for cleaner extraction:
  - `_final_answer_candidate(...)` to get strict vs relaxed candidates
  - `_strip_wrapping_pairs(...)` to remove wrappers like `<negative>`
  - `_extract_sentiment_label(...)` to safely recover `positive`/`negative` labels
- Added `format_violation` to each run record when fallback parsing is needed.
- Added `format_violation_count` to the summary CSV so prompt-format compliance is tracked separately from true parse failures.
- After rerunning on Hugging Face data (`samples-per-task=10`):
  - `Records with errors: 0`
  - `parse_fail_count: 0` across all task/technique rows
  - one format-violation case tracked under sentiment chain-of-thought

### Recent Implementation Update (Final HF Run + Sentiment Prompt Tweak)

- `experiments.py -> build_sentiment_prompt(...)` was tightened to reduce ambiguous outputs:
  - explicitly requires exactly one of:
    - `Final Answer: positive`
    - `Final Answer: negative`
  - explicitly says not to output angle brackets or both labels.
- Final assignment-size run completed with `--samples-per-task 20`:
  - `Loaded QA examples: 20`
  - `Loaded sentiment examples: 20`
  - `Collected records: 160`
  - `Records with errors: 0`
- Final metrics snapshot from `outputs/final_hf_20/summary_metrics.csv`:
  - QA best EM/F1: `zero_shot` (`EM=0.65`, `token_f1=0.7643`)
  - Sentiment best accuracy/macro-F1: tie `zero_shot` and `custom_variation` (`accuracy=0.90`, `macro_f1=0.899`)
  - `parse_fail_count=0` across all rows after the prompt tweak
  - remaining `format_violation_count=7`, all in sentiment `chain_of_thought`

### Recent Implementation Update (Part 3 Planning)

- Added a dedicated Part 3 report-prep TODO section to `PLAN.md`.
- Defined next concrete steps:
  - build report tables from `outputs/final_hf_20/summary_metrics.csv`
  - select 2-3 failure cases from `outputs/final_hf_20/experiment_runs.jsonl`
  - draft the 800-1200 word analysis structure

### Recent Implementation Update (Function-Level Documentation)

- Added concise multi-line docstrings across `experiments.py` for all core functions.
- Documented each function's purpose, inputs/outputs, and role in the Part 2 pipeline.
- Coverage includes:
  - normalization and metric helpers
  - prompt builders and parsing helpers
  - dataset loading/sampling
  - experiment execution loops
  - export and orchestration functions
- This improves readability for grading/review and makes report-to-code traceability clearer.
