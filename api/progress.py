"""
api/progress.py

Thread-safe progress event tracker and Server-Sent Events (SSE) broadcaster.
Supports live progress streaming and late SSE subscriber connections.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Set, Optional
from src.matching.progress import ProgressEvent

logger = logging.getLogger(__name__)


class JobProgressTracker:
    """
    Manages progress state history and active SSE queues for jobs.
    """

    def __init__(self):
        self._latest_event: Dict[str, ProgressEvent] = {}
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set main asyncio event loop for thread-safe queue publishing."""
        self._loop = loop

    def get_latest_event(self, job_id: str) -> Optional[ProgressEvent]:
        """Retrieve the latest progress event snapshot for a job."""
        return self._latest_event.get(job_id)

    def update_progress(self, job_id: str, event: ProgressEvent):
        """
        Thread-safe method called from model execution to publish progress events.
        """
        self._latest_event[job_id] = event

        # Notify active SSE subscribers on main event loop
        sub_set = self._subscribers.get(job_id, set())
        if sub_set and self._loop and self._loop.is_running():
            payload = format_sse_event("progress", event.to_dict())
            for q in list(sub_set):
                self._loop.call_soon_threadsafe(q.put_nowait, payload)

    def notify_job_finished(self, job_id: str, status: str, payload_data: Dict[str, Any]):
        """Notify SSE subscribers that job finished (completed or failed)."""
        sub_set = self._subscribers.get(job_id, set())
        if sub_set and self._loop and self._loop.is_running():
            event_type = "completed" if status == "completed" else "failed"
            payload = format_sse_event(event_type, payload_data)
            for q in list(sub_set):
                self._loop.call_soon_threadsafe(q.put_nowait, payload)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to SSE event stream for a specific job."""
        q = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        self._subscribers[job_id].add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        """Unsubscribe from SSE stream."""
        if job_id in self._subscribers:
            self._subscribers[job_id].discard(q)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    def cleanup_job(self, job_id: str):
        """Remove job history and subscribers."""
        self._latest_event.pop(job_id, None)
        self._subscribers.pop(job_id, None)


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format dictionary into standard SSE wire string."""
    json_str = json.dumps(data)
    return f"event: {event_type}\ndata: {json_str}\n\n"


# Global tracker instance
progress_tracker = JobProgressTracker()
