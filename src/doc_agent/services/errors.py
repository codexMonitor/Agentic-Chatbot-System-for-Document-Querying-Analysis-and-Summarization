import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Any failure talking to an LLM or embedding provider (including rate limits)."""


class IndexMismatchError(Exception):
    """The stored vectors were built with a different embedding model than the configured one."""


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int,
    what: str,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `func`, retrying rate-limit/server errors with exponential backoff.

    Every provider exception is re-raised as `LLMError`; the API key is never included.
    """
    attempt = 0
    while True:
        try:
            return func()
        except LLMError:
            raise
        except Exception as error:
            if _is_retryable(error) and attempt < max_retries:
                sleep(base_delay_s * 2**attempt)
                attempt += 1
                continue
            raise LLMError(f"{what} failed: {type(error).__name__}: {error}") from error
