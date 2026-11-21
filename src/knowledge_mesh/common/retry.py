"""Retry with exponential backoff and a clear terminal failure."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .errors import TransientError

T = TypeVar("T")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff: float = 0.5
    max_backoff: float = 10.0
    multiplier: float = 2.0

    def delay(self, attempt: int) -> float:
        base = min(self.max_backoff, self.initial_backoff * (self.multiplier ** (attempt - 1)))
        return base * (0.5 + random.random() / 2)


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    on_retry: Callable[[int, BaseException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempt = 1
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                raise
            if on_retry:
                on_retry(attempt, exc)
            log.warning("retrying after failure", extra={"attempt": attempt, "error": type(exc).__name__})
            sleep(policy.delay(attempt))
            attempt += 1
