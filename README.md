My NLP course at UTSA

Instructor: Dr. Rad

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai
```

Set your API key and run:

```bash
export COURSE_LLM_API_KEY="<your_key>"
python api_basics.py --temperature 0.7 --max-tokens 200
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
