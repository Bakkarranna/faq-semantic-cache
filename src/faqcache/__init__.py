"""Semantic caching for an FAQ support bot.

A repeat question is answered from a cached answer when its meaning matches a
stored question closely enough; everything else goes to an LLM.
"""

from faqcache.cache import CacheDecision, SemanticCache
from faqcache.config import Settings

__all__ = ["CacheDecision", "SemanticCache", "Settings"]
__version__ = "1.0.0"
