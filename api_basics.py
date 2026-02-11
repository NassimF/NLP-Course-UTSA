#!/usr/bin/env python3
"""
api_basics.py — CS6263 Assignment 1 (Part 1: API Basics)

API Provider Used: Course Llama Server (OpenAI-compatible API)

What this script includes (per Part 1 requirements):
- query_llm(prompt, **kwargs) wrapper with temperature + max_tokens
- Proper error handling (connection errors, rate limits, auth failures, timeouts)
- Retry mechanism with exponential backoff for transient failures
- main() demonstrating at least 3 prompts and printing responses
"""

import os
import sys
import time
import random
import traceback
import re
import argparse
from typing import Optional, Dict, Any

from openai import OpenAI


# -----------------------------
# Configuration (Course Server)
# -----------------------------


COURSE_LLM_BASE_URL = os.getenv("COURSE_LLM_BASE_URL", "http://10.246.100.230/v1")
COURSE_LLM_API_KEY  = os.getenv("COURSE_LLM_API_KEY")
COURSE_LLM_MODEL    = os.getenv("COURSE_LLM_MODEL","Llama-3.1-70B-Instruct-custom")
if not COURSE_LLM_API_KEY:
    raise RuntimeError("COURSE_LLM_API_KEY is not set. Export it in your shell before running.")



# Optional: request timeout (seconds)
COURSE_LLM_TIMEOUT = float(os.getenv("COURSE_LLM_TIMEOUT", "120"))

# System prompt used for all requests.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer directly. Do not include chain-of-thought "
    "or <think> tags. If the user requests JSON, output only valid JSON. Follow "
    "format instructions exactly."
)

# Create the OpenAI client configured for the course server.
# Timeout is set at the client level (simple, global default).
client = OpenAI(
    base_url=COURSE_LLM_BASE_URL,
    api_key=COURSE_LLM_API_KEY,
    max_retries=0,      # disable SDK retries; we handle retries ourselves
)



# -----------------------------
# Error Classification Helpers
# -----------------------------
def _safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _classify_error(exc: Exception) -> Dict[str, Any]:
    """
    Try to classify common API failures into:
    - auth_failure (do not retry)
    - rate_limited (retry)
    - timeout (retry)
    - connection (retry)
    - server_error (retry)
    - bad_request (do not retry)
    - unknown (retry a limited amount)
    """
    msg = str(exc).lower()

    # Some OpenAI errors carry HTTP status on `.status_code` or `.response.status_code`
    status_code = _safe_getattr(exc, "status_code", None)
    resp = _safe_getattr(exc, "response", None)
    if status_code is None and resp is not None:
        status_code = _safe_getattr(resp, "status_code", None)

    # Authentication failures (401/403 or obvious message)
    if status_code in (401, 403) or "authentication" in msg or "api key" in msg or "unauthorized" in msg:
        return {"type": "auth_failure", "retry": False, "status_code": status_code}

    # Rate limiting (429 or message)
    if status_code == 429 or "rate limit" in msg or "too many requests" in msg:
        return {"type": "rate_limited", "retry": True, "status_code": status_code}

    # Timeouts
    if "timeout" in msg or "timed out" in msg:
        return {"type": "timeout", "retry": True, "status_code": status_code}

    # Connection issues
    if "connection" in msg or "connection error" in msg or "failed to connect" in msg or "network" in msg:
        return {"type": "connection", "retry": True, "status_code": status_code}

    # Server errors (5xx)
    if status_code is not None and 500 <= int(status_code) <= 599:
        return {"type": "server_error", "retry": True, "status_code": status_code}

    # Bad requests (400/404/422): usually prompt/model config issues
    if status_code is not None and int(status_code) in (400, 404, 409, 422):
        return {"type": "bad_request", "retry": False, "status_code": status_code}

    # Fall back
    return {"type": "unknown", "retry": True, "status_code": status_code}


def _exponential_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    """
    Exponential backoff with jitter:
      attempt=0 -> ~1s
      attempt=1 -> ~2s
      attempt=2 -> ~4s
    Capped to avoid unbounded sleep.
    """
    exp = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, 0.25 * exp)
    return exp + jitter


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit before the final answer."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


# -----------------------------
# Part 1 Required Function
# -----------------------------
def query_llm(prompt: str, **kwargs) -> str:
    """
    Sends a prompt to the LLM and returns the response text.

    Required params per assignment:
      - temperature
      - max_tokens

    Also includes:
      - error handling (rate limits, auth failures, timeouts, connection)
      - retry with exponential backoff (transient errors only)
    """
    temperature = float(kwargs.get("temperature", 0.7))
    max_tokens = int(kwargs.get("max_tokens", 100))
    model = kwargs.get("model", None)
    retries = int(kwargs.get("retries", 5))
    timeout = kwargs.get("timeout", None)

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    use_model = model or COURSE_LLM_MODEL
    use_timeout = COURSE_LLM_TIMEOUT if timeout is None else float(timeout)

    last_exc: Optional[Exception] = None
    last_info: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        try:
            # OpenAI-compatible Chat Completions call.
            # Works with many OpenAI-compatible servers, including course Llama suggests.
            response = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},#skip the chain of thought printing
                {"role": "user", "content": prompt},
                ],
            temperature=temperature,#randomness
            top_p=1,#nucleus sampling, higher-> more variety
            max_tokens=max_tokens,
            frequency_penalty=0,#no penalty for repeating tokens based on frequency
            presence_penalty=0,# no penalty for repeating topics
            timeout=use_timeout,
            )

            # Extract the text content safely
            text = ""
            if response and response.choices:
                msg = response.choices[0].message
                text = (msg.content or "").strip()

            return _strip_think(text)

        except Exception as exc:
            last_exc = exc
            info = _classify_error(exc)
            last_info = info

            # Non-retryable: fail fast with a clear message
            if not info["retry"]:
                raise RuntimeError(
                    f"Non-retryable error ({info['type']}, status={info.get('status_code')}): {exc}"
                ) from exc

            # Retryable but out of attempts
            if attempt >= retries:
                break

            # Sleep before retrying
            sleep_s = _exponential_backoff_seconds(attempt)
            print(
                f"[WARN] Retryable error ({info['type']}, status={info.get('status_code')}). "
                f"Attempt {attempt+1}/{retries}. Sleeping {sleep_s:.2f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    # If we got here, retries exhausted
    if last_info and last_info.get("status_code") == 503:
        raise RuntimeError(
            "Failed after "
            f"{retries+1} attempts. Server returned 503 Service Unavailable "
            "(timeout/overload). Client-side timeouts won't fix this; "
            "try later or use a smaller model. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise RuntimeError(f"Failed after {retries+1} attempts. Last error: {last_exc}") from last_exc


# -----------------------------
# Demo (Part 1 requirement)
# -----------------------------
def main() -> None:
    
    parser = argparse.ArgumentParser(description="API Basics demo client.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens for responses.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--retries", type=int, default=5, help="Retry attempts for transient failures.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout override.")
    args = parser.parse_args()

    print("== Part 1 Demo: API Basics ==")
    print(f"Base URL: {COURSE_LLM_BASE_URL}")
    print(f"Model:    {args.model or COURSE_LLM_MODEL}")
    print("Note: Set COURSE_LLM_BASE_URL / COURSE_LLM_API_KEY / COURSE_LLM_MODEL as env vars.\n")

    prompts = [
        "Write a one-sentence explanation of what prompt engineering means.",
        "Give me 3 bullet points on why error handling matters when calling an API.",
        "Return a JSON object with keys: task, difficulty, tip. Keep values short.",
    ]

    # parameters
    temperature_default = args.temperature
    temperature_formatting = 0.0
    max_tokens = args.max_tokens
    inter_prompt_sleep_s = 1.5

    for i, p in enumerate(prompts, start=1):
        print(f"--- Prompt #{i} ---")
        print(p)
        print("\n--- Response ---")
        temp = temperature_default if i == 1 else temperature_formatting
        try:
            out = query_llm(
                p,
                temperature=temp,
                max_tokens=max_tokens,
                model=args.model,
                retries=args.retries,
                timeout=args.timeout,
            )
            print(out if out else "[Empty response]")
        except Exception as e:
            print("[ERROR] query_llm failed.")
            print(str(e))
            #traceback for debugging 
            traceback.print_exc()
        print("\n")
        time.sleep(inter_prompt_sleep_s)


if __name__ == "__main__":
    main()
