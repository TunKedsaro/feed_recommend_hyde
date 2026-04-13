# functions/utils/llm_client.py
from __future__ import annotations

"""
functions.utils.llm_client

Strict JSON-only LLM client utilities for HyDE query generation.

This module provides a **deterministic, debuggable wrapper** around the Google
GenAI (Gemini) SDK that enforces a JSON-object output contract.

Design goals
------------
- JSON-only output contract (no free text accepted)
- Deterministic retry behavior:
  - Retries ONLY on JSON parsing/extraction failures (ValueError)
  - Exponential backoff with bounded sleep
- Best-effort extraction across SDK versions:
  - Prefer resp.parsed (structured)
  - Fall back to candidate parts text
  - Fall back to resp.text
- Best-effort telemetry:
  - latency (perf_counter)
  - token usage (usage_metadata) when available
- No business logic:
  - Prompt templates / schema decisions live outside this module

POC vs Production note
----------------------
- In this POC, Gemini is used ONLY in:
  - batch HyDE generation (pipeline_1_user_hyde.py)
  - online single-user refresh (pipeline_2_user_hyde_refresh.py)
- Online serving (pipeline_3_online_retrieval.py) MUST remain LLM-free.

Public API
----------
- GeminiJsonClient: strict JSON-only generate_json(prompt) -> Dict[str, Any]
- build_llm_client(): stable factory used by pipelines/tests (easy monkeypatch)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.functions.utils.logging import get_logger
from src.functions.utils.cost_logger import append_cost_log

logger = get_logger(__name__)


# =============================================================================
# YAML utilities
# =============================================================================
def _load_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML file into a dict.

    Contract:
    - If path is empty => {}
    - File must exist (fail-fast) when path is provided
    - Top-level must be a dict

    Rationale:
    - HyDE/LLM configs should be explicit; failing fast avoids silent misconfigs.
    """
    import yaml

    if not path:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML must map to a dict: {path}")

    return data


# =============================================================================
# JSON extraction / safe repair helpers (deterministic; no hallucination)
# =============================================================================
def _try_autoclose_json(text: str) -> Optional[str]:
    """
    Attempt a **safe auto-close** for truncated JSON objects.

    What this DOES:
    - Counts unmatched `{` and `[` (outside of string literals)
    - Appends missing closing `]` / `}`

    What this DOES NOT:
    - Invent keys/values
    - Modify existing content
    - Repair broken/unclosed strings

    Returns:
      Repaired JSON string if it looks safe, else None.
    """
    s = (text or "").strip()

    # Only attempt to repair JSON objects. (HyDE bundle schemas are objects.)
    if not s.startswith("{"):
        return None

    open_curly = 0
    open_square = 0
    in_str = False
    escape = False

    for ch in s:
        if escape:
            escape = False
            continue

        if ch == "\\" and in_str:
            escape = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        # Ignore structure tokens inside string literals.
        if in_str:
            continue

        if ch == "{":
            open_curly += 1
        elif ch == "}":
            open_curly = max(0, open_curly - 1)
        elif ch == "[":
            open_square += 1
        elif ch == "]":
            open_square = max(0, open_square - 1)

    # Unsafe to repair if a string literal is still open.
    if in_str:
        return None

    # Already balanced.
    if open_curly == 0 and open_square == 0:
        return s

    # Close arrays first, then objects (mirrors typical nesting).
    return s + ("]" * open_square) + ("}" * open_curly)


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first complete JSON object from a mixed or wrapped response.

    Used when:
    - The model wraps JSON with explanation text
    - The SDK does not expose parsed output (or parsing failed)

    Raises:
      ValueError if a complete JSON object cannot be found.
    """
    if not text:
        raise ValueError("Empty LLM response")

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("JSON object appears truncated (no matching '}')")


def _extract_json_from_response(resp: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from a google-genai response.

    Priority order:
    1) resp.parsed (structured output; preferred)
    2) candidates[0].content.parts[*].text (raw JSON text)
    3) None (caller may try resp.text as last resort)

    Returns:
      dict if found and parsed as dict; otherwise None.
    """
    # 1) Structured parsed output (best-case)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    # 2) Candidate text parts (SDK-version tolerant)
    candidates = getattr(resp, "candidates", None)
    if isinstance(candidates, list) and candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None)

        if isinstance(parts, list):
            for part in parts:
                txt = getattr(part, "text", None)
                if not isinstance(txt, str) or not txt.strip():
                    continue

                raw = txt.strip()

                # Try direct parse
                try:
                    obj = json.loads(raw)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    pass

                # Try extracting first JSON object
                try:
                    obj = json.loads(_extract_first_json_object(raw))
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    pass

                # Try safe auto-close
                repaired = _try_autoclose_json(raw)
                if repaired:
                    try:
                        obj = json.loads(repaired)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        pass

    return None


# =============================================================================
# Telemetry helpers (best-effort; SDK-version tolerant)
# =============================================================================
def _extract_token_usage(resp: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Best-effort token usage extraction from google-genai response.

    Depending on SDK version/model, response may include `usage_metadata` with fields:
    - prompt_token_count
    - candidates_token_count

    Returns:
      (input_tokens, output_tokens) as ints or None when unavailable.
    """
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return None, None

    in_tok = getattr(usage, "prompt_token_count", None)
    out_tok = getattr(usage, "candidates_token_count", None)

    try:
        in_tok_i = int(in_tok) if in_tok is not None else None
    except Exception:
        in_tok_i = None

    try:
        out_tok_i = int(out_tok) if out_tok is not None else None
    except Exception:
        out_tok_i = None

    return in_tok_i, out_tok_i


def _fmt_tok(x: Optional[int]) -> str:
    """Pretty-format token usage for logs."""
    return str(x) if x is not None else "NA"

def _estimate_llm_cost_usd(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    model_name: str,
) -> float:
    """
    Temporary cost estimator.
    Replace these rates with the real Gemini pricing you use.
    """
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    # TODO: replace with real pricing
    pricing = {
        "gemini-3.1-flash-lite-preview": {
            "input_per_1k": 0.0001,
            "output_per_1k": 0.0002,
        }
    }

    model_price = pricing.get(
        model_name,
        {"input_per_1k": 0.0001, "output_per_1k": 0.0002},
    )

    input_cost = (input_tokens / 1000) * model_price["input_per_1k"]
    output_cost = (output_tokens / 1000) * model_price["output_per_1k"]

    return input_cost + output_cost
# =============================================================================
# Gemini JSON client
# =============================================================================
@dataclass
class GeminiJsonClient:
    """
    Strict JSON-only Gemini client with deterministic retry behavior.

    Retry policy
    ------------
    Retries ONLY on ValueError (parse/extraction failure). This prevents retries on:
    - networking errors
    - SDK errors
    - authorization errors
    Those should fail fast so operators can see the real issue.

    Notes:
    - retry_limit = N => total attempts = N + 1
    - Exponential backoff: 1s -> 8s (bounded)

    This client MUST NOT:
    - contain prompt logic
    - contain business rules
    """

    api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout_s: int = 60  # currently best-effort; SDK may not expose timeout per request
    retry_limit: int = 2  # number of retries (not total attempts)

    def __post_init__(self) -> None:
        if not str(self.api_key).strip():
            raise ValueError("Missing api_key for GeminiJsonClient")

        # Lazy import to keep serving paths import-safe when LLM isn't needed.
        from google import genai  # type: ignore

        # Initialize Gemini client once (reuse across calls for performance).
        self._client = genai.Client(api_key=str(self.api_key).strip())

        # Build a retry decorator dynamically so policy is config-driven.
        attempts = max(1, int(self.retry_limit) + 1)
        self._retry_decorator = retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(ValueError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    # def generate_json(self, prompt: str) -> Dict[str, Any]:
    def generate_json(
    self,
    prompt: str,
    extra_log: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
        """
        Generate a JSON object from the LLM.

        Contract:
        - Returns a dict (JSON object) on success.
        - Raises ValueError if JSON cannot be extracted after retries.

        Telemetry:
        - Logs per-attempt latency and best-effort token usage.
        - On failures, logs only a bounded prefix of the raw text to avoid huge logs.

        IMPORTANT:
        - This method does not modify JSON content.
        - The only "repair" is safe auto-close of braces/brackets when JSON is truncated.
        """
        attempt_no = {"n": 0}
        # print(f"attempt_no -> {attempt_no}")

        @self._retry_decorator
        def _call_once() -> Dict[str, Any]:
            attempt_no["n"] += 1

            # Local import to keep module import-time light.
            from google.genai import types  # type: ignore

            # Use perf_counter for monotonic, stable latency measurement.
            t0 = time.perf_counter()

            # print(f"prompt -> {prompt}")
            # print(f"model -> {self.model_name}")
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=float(self.temperature),
                    max_output_tokens=int(self.max_output_tokens),
                    # Strong hint to the model + SDK that response must be JSON.
                    response_mime_type="application/json",
                ),
            )

            # resp = self._client.models.generate_content(
            #     model=self.model_name,
            #     contents = prompt,
            #     config=types.GenerateContentConfig(
            #         temperature=float(self.temperature),
            #         max_output_tokens=int(self.max_output_tokens),
            #         response_mime_type="application/json",
            #         thinking_config=types.ThinkingConfig(thinking_level="low")
            #     ),
            # )

            # print(f"prompt -> {prompt}")
            # print(f"resp -> {resp}")
            # print("="*100)

            latency_s = time.perf_counter() - t0
            in_tok, out_tok = _extract_token_usage(resp)

            estimated_cost_usd = _estimate_llm_cost_usd(
                input_tokens=in_tok,
                output_tokens=out_tok,
                model_name=self.model_name,
            )

            payload = {
                "event_type": "llm",
                "model_name": self.model_name,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": round(latency_s, 4),
                "estimated_cost_usd": estimated_cost_usd,
            }

            if extra_log:
                payload.update(extra_log)

            append_cost_log(payload)


            # Prefer structured extraction (SDK-parsed or candidate parts).
            structured = _extract_json_from_response(resp)
            if structured is not None:
                logger.info(
                    "LLM call done | attempt=%d | latency=%.3fs | in_tokens=%s | out_tokens=%s | model=%s | status=ok",
                    attempt_no["n"],
                    latency_s,
                    _fmt_tok(in_tok),
                    _fmt_tok(out_tok),
                    self.model_name,
                )
                return structured

            # Last resort: raw text extraction (SDK differences).
            raw = (getattr(resp, "text", None) or "").strip()

            # Extract the first JSON object (if wrapped).
            try:
                obj = json.loads(_extract_first_json_object(raw))
                if not isinstance(obj, dict):
                    raise ValueError("LLM returned JSON but not an object/dict")
                logger.info(
                    "LLM call done | attempt=%d | latency=%.3fs | in_tokens=%s | out_tokens=%s | model=%s | status=ok_text_fallback",
                    attempt_no["n"],
                    latency_s,
                    _fmt_tok(in_tok),
                    _fmt_tok(out_tok),
                    self.model_name,
                )
                return obj
            except Exception:
                pass

            # Safe auto-close (only if we can do it without changing content).
            repaired = _try_autoclose_json(raw)
            if repaired:
                try:
                    obj = json.loads(repaired)
                    if not isinstance(obj, dict):
                        raise ValueError("LLM returned JSON but not an object/dict")
                    logger.info(
                        "LLM call done | attempt=%d | latency=%.3fs | in_tokens=%s | out_tokens=%s | model=%s | status=ok_autoclose",
                        attempt_no["n"],
                        latency_s,
                        _fmt_tok(in_tok),
                        _fmt_tok(out_tok),
                        self.model_name,
                    )
                    return obj
                except Exception:
                    pass

            # Log a bounded preview for debugging (avoid huge payloads).
            preview = raw[:200].replace("\n", "\\n") if raw else ""
            logger.warning(
                "LLM JSON parse failed (will retry) | attempt=%d | latency=%.3fs | in_tokens=%s | out_tokens=%s | model=%s | first_200=%s",
                attempt_no["n"],
                latency_s,
                _fmt_tok(in_tok),
                _fmt_tok(out_tok),
                self.model_name,
                preview,
            )
            raise ValueError("LLM returned malformed or non-extractable JSON")

        return _call_once()


# =============================================================================
# Factories
# =============================================================================
def build_llm_client_from_yaml(
    parameters_path: str = "parameters/parameters.yaml"
) -> GeminiJsonClient:
    """
    Factory for GeminiJsonClient using repo YAML configuration.

    Reads
    -----
    - parameters.yaml:
        llm.model_name
        llm.temperature
        llm.max_output_tokens
        llm.timeout_s              (optional; best-effort)
        hyde.retry_limit
    - credentials.yaml OR env var:
        GOOGLE_API_KEY

    Precedence for GOOGLE_API_KEY:
      env var > credentials.yaml > ""

    Returns:
      GeminiJsonClient
    """
    params = _load_yaml(parameters_path)
    # creds = _load_yaml(credentials_path)

    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or str((creds.get("GOOGLE_API_KEY") if isinstance(creds, dict) else "") or "").strip()
    )

    llm_cfg = params.get("llm", {}) if isinstance(params, dict) else {}
    hyde_cfg = params.get("hyde", {}) if isinstance(params, dict) else {}

    return GeminiJsonClient(
        api_key=str(api_key).strip(),
        model_name=str(llm_cfg.get("model_name", "gemini-2.5-flash")),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_output_tokens=int(llm_cfg.get("max_output_tokens", 1024)),
        timeout_s=int(llm_cfg.get("timeout_s", 60)),
        retry_limit=int(hyde_cfg.get("retry_limit", 2)),
    )


def build_llm_client(
    parameters_path: str = "parameters/parameters.yaml",
    credentials_path: str = "parameters/credentials.yaml",
) -> GeminiJsonClient:
    """
    Stable alias for the YAML-based client factory.

    Why it exists:
    - Tests can monkeypatch this symbol as the single "LLM entrypoint".
    - Avoids refactor churn if the underlying factory name changes later.
    """
    return build_llm_client_from_yaml(
        parameters_path=parameters_path,
        credentials_path=credentials_path,
    )


__all__ = [
    "GeminiJsonClient",
    "build_llm_client_from_yaml",
    "build_llm_client",
]
