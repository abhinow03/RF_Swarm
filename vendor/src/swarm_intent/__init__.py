"""swarm_intent — UAV swarm formation classification + LLM tactical reasoning.

Single source of truth for code that was previously duplicated across three
notebooks. See MIGRATION_GUIDE.md for the layout and how to run things.
"""
from .config import Config, formation_names, BASE_FORMATIONS, TRANSITION_CLASS

__all__ = ["Config", "formation_names", "BASE_FORMATIONS", "TRANSITION_CLASS"]
__version__ = "0.1.0"
