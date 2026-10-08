"""Sliding-window rate limiter (adapted from app/core/guardrails.py)."""
import time


class RateLimiter:
    """At most `max_requests` per `window` seconds for each user."""

    def __init__(self, max_requests=20, window=60.0, clock=time.time):
        self.max_requests = max_requests
        self.window = window
        self.clock = clock
        self.requests = {}

    def allow(self, user_id: str) -> bool:
        now = self.clock()
        recent = [t for t in self.requests.get(user_id, []) if now - t < self.window]
        self.requests[user_id] = recent
        if len(recent) > self.max_requests:
            return False
        recent.append(now)
        return True
