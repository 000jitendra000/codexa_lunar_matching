"""
src/utils/memory_debug.py

Optional local memory diagnostics helper behind LUNAR_MEMORY_DEBUG=1.
Measures process Resident Set Size (RSS) in MB at key execution checkpoints.
Disabled by default with zero production overhead.
"""

import os
import logging

logger = logging.getLogger(__name__)


def is_memory_debug_enabled() -> bool:
    """Return True if LUNAR_MEMORY_DEBUG environment variable is set to 1, true, or yes."""
    return os.getenv("LUNAR_MEMORY_DEBUG", "0").lower() in ("1", "true", "yes")


def get_rss_mb() -> float:
    """Return process Resident Set Size (RSS) in Megabytes if psutil is available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def log_memory_stage(stage: str) -> float:
    """
    Log process RSS at a given execution checkpoint if LUNAR_MEMORY_DEBUG is enabled.
    
    Returns:
        float: Process RSS in MB (or 0.0 if disabled).
    """
    if is_memory_debug_enabled():
        rss = get_rss_mb()
        logger.info("[MEMORY_DEBUG] RSS at %-35s: %8.2f MB", stage, rss)
        return rss
    return 0.0
