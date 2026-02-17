# Repository Guidelines

## Project Structure & Module Organization
This repository is currently centered on two executable scripts:
- `api_basics.py`: main assignment client for OpenAI-compatible course APIs (request wrapper, retries, CLI demo).
- `experiments.py`: Part 2 experiment runner for QA + sentiment prompt-technique comparisons.
- `README.md`: brief course context and networking note.
- `Assignment1_LLM_APIs_and_Prompting_Accessible-v1.docx`: assignment handout/reference.

Keep new Python modules in the repository root unless a `src/` layout is introduced. If tests are added, place them in `tests/`.

## Build, Test, and Development Commands
Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai datasets
```

Run the scripts:

```bash
export COURSE_LLM_API_KEY="<your_key>"
python api_basics.py --temperature 0.7 --max-tokens 200
python experiments.py --samples-per-task 20 --output-dir outputs/final_hf_20
```

Optional networking fix on the DGX environment:

```bash
export NO_PROXY="10.246.100.230,localhost,127.0.0.1"
```

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables and `UPPER_SNAKE_CASE` for module constants (e.g., `COURSE_LLM_BASE_URL`).
- Prefer small helper functions for error handling and retries.
- Add type hints for public functions and keep docstrings concise and purpose-driven.

## Testing Guidelines
There is no automated test suite yet. Minimum requirement for contributions is a manual smoke test:

```bash
python api_basics.py --model "Llama-3.1-70B-Instruct-custom"
python experiments.py --samples-per-task 1 --use-local-smoke-data --output-dir outputs/smoke_validation_local
```

When adding tests, use `pytest` with files named `tests/test_<module>.py`; focus on retry behavior, error classification, and argument parsing.

## Commit & Pull Request Guidelines
Recent commits use short, imperative messages (for example: `add argparse`, `minor cleanup`). Continue this style:
- Format: `<verb> <what changed>` (lowercase is acceptable in this repo).
- Keep commits focused and logically separated.

For pull requests:
- Include a clear summary, linked issue/assignment requirement, and key command(s) used for verification.
- Paste relevant terminal output for behavior changes.
- Keep diffs small and explain any API/network assumptions.
