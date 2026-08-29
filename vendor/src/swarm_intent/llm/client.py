"""
Provider-agnostic LLM client.

Replaces the copy-pasted ``call_llm`` in every notebook with a single class that
can talk to:
  - Groq (hosted, for the prompt-engineered baseline)
  - a local HuggingFace model (your QLoRA-fine-tuned Qwen/Mistral)

This lets the rest of the codebase (pipeline, eval) stay identical whether you
are using the API baseline or your fine-tuned model — just swap the client.

Robustness fixes over the original: HTTP error checking, retry with backoff,
and tolerant JSON extraction.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _generate_one_with_retry(client: "LLMClient", prompt: str, system_prompt: Optional[str]) -> str:
    """Same 3-attempt/exponential-backoff shape as LLMClient.complete(), but
    returns raw text (never raises) so a single flaky/rate-limited request in a
    threaded batch can't take down the whole batch. Total failure returns ""
    -- fails JSON parsing downstream, which complete_batch's caller already
    treats as a normal per-row teacher fallback."""
    for attempt in range(3):
        try:
            return client.generate(prompt, system_prompt=system_prompt)
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2 ** attempt)
    return ""


class _ThreadedBatchMixin:
    """Mixin for HTTP-API clients (network-bound, GIL-released during I/O):
    runs generate() concurrently across a thread pool instead of LocalHFClient's
    GPU-batched forward pass. batch_size doubles as max_workers -- keep it
    within whatever concurrent-request limit the provider actually allows."""

    def generate_batch(self, prompts: list[str], batch_size: int = 8,
                       system_prompt: Optional[str] = None) -> list[str]:
        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            futures = [ex.submit(_generate_one_with_retry, self, p, system_prompt) for p in prompts]
            return [f.result() for f in futures]


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, fences or not."""
    # Reasoning models (qwen3, deepseek-r1) prepend a <think>...</think> block;
    # drop it first so a stray brace inside the thinking prose can't poison the
    # greedy {.*} fallback search below.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _parse_llm_response(text: str) -> dict:
    """_extract_json wrapped in the same tolerant try/except complete() uses for
    its final attempt -- shared so batched paths (no retry/backoff, since local
    generation isn't network-flaky) parse identically to the single-call path."""
    try:
        return _extract_json(text)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed"}


class LLMClient:
    """Base interface. Implement ``generate(prompt) -> str``."""

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        for attempt in range(3):
            try:
                return _extract_json(self.generate(prompt, system_prompt=system_prompt))
            except json.JSONDecodeError:
                if attempt == 2:
                    return {"error": "JSON parse failed"}
            except Exception as e:  # network etc.
                if attempt == 2:
                    return {"error": str(e)}
                time.sleep(2 ** attempt)
        return {"error": "unreachable"}

    def complete_batch(self, prompts: list[str], batch_size: int = 8,
                       system_prompt: Optional[str] = None) -> list[dict]:
        """Default: no batching support, falls back to sequential complete().
        Clients that can batch (LocalHFClient) override generate_batch()."""
        try:
            raw = self.generate_batch(prompts, batch_size=batch_size, system_prompt=system_prompt)
            return [_parse_llm_response(r) for r in raw]
        except NotImplementedError:
            return [self.complete(p, system_prompt=system_prompt) for p in prompts]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_batch(self, prompts: list[str],
                       system_prompt: Optional[str] = None) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError


class GroqClient(_ThreadedBatchMixin, LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 api_key: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY (env var) or pass api_key=...")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": self.temperature, "max_tokens": self.max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class NvidiaClient(_ThreadedBatchMixin, LLMClient):
    """NVIDIA NIM-hosted teacher (build.nvidia.com), OpenAI-compatible endpoint.
    Replaces GroqClient as the Phase 1 corpus-generation teacher (AUDIT.md V5
    Phase 1 step 0) -- same request/response shape, so build_sft_dataset.py's
    gold_assessment()/teacher.complete() call sites are unchanged."""

    def __init__(self, model: str = "nvidia/nemotron-3-super-120b-a12b",
                 api_key: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("Set NVIDIA_API_KEY (env var) or pass api_key=...")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
        resp = requests.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": self.temperature, "max_tokens": self.max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# AUDIT.md pre-training-gate step 1: the advisory-only judge used across every
# evaluate_llm(judge_client=...) call site was GroqClient(llama-3.3-70b-versatile).
# Retired because (a) Groq's earlier retirement (Phase 1 step 0) was due to an
# unverifiable key-rotation/exposure concern that was never actually confirmed
# safe -- reusing a fresh Groq key for judge purposes would quietly reintroduce
# that same unresolved risk, and (b) GROQ_API_KEY was never set in this
# environment to begin with. Routed through NvidiaClient instead, since that
# infrastructure/credential is already verified working (Phase 1 corpus
# generation). JUDGE_MODEL is deliberately NOT nemotron -- nemotron is the
# Phase 1 teacher that authored v5-a's own training corpus; using it as judge
# too would contaminate the score with the exact prose/reasoning patterns v5-a
# was fine-tuned to imitate (a variant of the self-grading bug
# src/swarm_intent/llm/evaluate.py's own docstring already documents and fixed
# once). Verified live (2026-08-11) via the NVIDIA_API_KEY already exported for
# Phase 1: a real generate() call succeeded.
JUDGE_MODEL = "meta/llama-3.1-70b-instruct"


def default_judge_client() -> Optional["NvidiaClient"]:
    """The advisory-only judge every eval script passes as evaluate_llm's
    judge_client. Returns None if NVIDIA_API_KEY isn't set -- every call site
    already handles a None judge gracefully (this is by design: the judge is
    advisory-only, never load-bearing for headline/objective metrics)."""
    if not os.environ.get("NVIDIA_API_KEY"):
        return None
    return NvidiaClient(model=JUDGE_MODEL)


class LocalHFClient(LLMClient):
    """Wraps a local (optionally QLoRA-fine-tuned) HuggingFace causal LM.

    Lazy-imports transformers so the rest of the package works without it.
    """

    def __init__(self, model_path: str, adapter_path: Optional[str] = None,
                 temperature: float = 0.3, max_new_tokens: int = 512, load_4bit: bool = True,
                 system_prompt: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.tok = AutoTokenizer.from_pretrained(model_path)
        quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
                 if load_4bit else None)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant, device_map="auto",
            torch_dtype=torch.float16,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import torch
        effective_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages = ([{"role": "system", "content": effective_system_prompt}]
                   if effective_system_prompt else [])
        messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(self.model.device)
        sample_kwargs = ({"do_sample": True, "temperature": self.temperature}
                         if self.temperature > 0 else {"do_sample": False})
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **sample_kwargs)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate_batch(self, prompts: list[str], batch_size: int = 1,
                       system_prompt: Optional[str] = None) -> list[str]:
        """Batched generation: LEFT-padding (required for correct batched causal-LM
        generation -- all sequences must end at the same position so the new-token
        start offset is identical across the batch), prompts sorted by tokenized
        length before chunking into `batch_size` groups to cut padding waste, then
        restored to the caller's original order. Same message-building and
        sampling settings as generate(), so a batch_size=1 call reproduces
        generate()'s output distribution exactly (same seed/temperature aside).

        system_prompt: overrides self.system_prompt for this call only, same
        per-call-override convention as generate() (used by pipeline_v2.py's
        Layer 1, which reuses one base-model client for both rules_in_prompt's
        RULES.txt system prompt and its own narrative-only system prompt)."""
        import torch

        effective_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        old_padding_side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            all_messages = []
            for prompt in prompts:
                messages = ([{"role": "system", "content": effective_system_prompt}]
                           if effective_system_prompt else [])
                messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
                all_messages.append(messages)

            # Sort by prompt length (not exact token count, but effectively monotonic
            # with it) so each chunk pads to a similar length, not the global max.
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
            results: list[Optional[str]] = [None] * len(prompts)
            sample_kwargs = ({"do_sample": True, "temperature": self.temperature}
                             if self.temperature > 0 else {"do_sample": False})

            for start in range(0, len(order), batch_size):
                chunk_idx = order[start:start + batch_size]
                chunk_messages = [all_messages[i] for i in chunk_idx]
                enc = self.tok.apply_chat_template(
                    chunk_messages, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt", padding=True).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                              pad_token_id=self.tok.eos_token_id, **sample_kwargs)
                prompt_len = enc["input_ids"].shape[1]  # identical for every row: left-padded to one length
                for row, i in enumerate(chunk_idx):
                    results[i] = self.tok.decode(out[row][prompt_len:], skip_special_tokens=True)

            return results
        finally:
            self.tok.padding_side = old_padding_side
