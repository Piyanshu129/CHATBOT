"""Simple in-memory cache manager."""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages in-memory caching with TTL (Time To Live).
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized CacheManager")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry["expiry"]:
                logger.debug(f"Cache hit for key: {key}")
                return entry["value"]
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache with TTL."""
        self._cache[key] = {"value": value, "expiry": time.time() + ttl_seconds}
        logger.debug(f"Cache set for key: {key} (TTL: {ttl_seconds}s)")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
