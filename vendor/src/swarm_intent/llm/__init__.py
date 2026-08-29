"""LLM interpretation layer: clients, prompts, pipeline, evaluation."""
from .client import LLMClient, GroqClient, LocalHFClient

__all__ = ["LLMClient", "GroqClient", "LocalHFClient"]
