import os
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MIN_SECRET_LENGTH = 6


def mask_secrets(text: str) -> str:
    """Replace the value of every configured `*API_KEY` environment variable with `***`."""
    for name, value in os.environ.items():
        if name.upper().endswith("API_KEY") and len(value.strip()) >= _MIN_SECRET_LENGTH:
            text = text.replace(value.strip(), "***")
    return text


class LLMError(Exception):
    """Any failure talking to an LLM or embedding provider (including rate limits).

    `retryable` marks transient failures (rate limits, temporary server errors) that a caller
    may retry; authentication and configuration errors leave it False. The message is masked,
    so a key echoed back by an SDK never reaches logs or the console.
    """

    def __init__(self, message: str = "", *, retryable: bool = False) -> None:
        super().__init__(mask_secrets(message))
        self.retryable = retryable


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
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Run `func`, retrying rate-limit/server errors with exponential backoff.

    Every provider exception is re-raised as `LLMError` (flagged `retryable` when it was a
    transient one); an `LLMError` raised by `func` is retried only if it is flagged `retryable`.
    The API key is never included.
    """
    attempt = 0
    while True:
        try:
            return func()
        except LLMError as error:
            if not (error.retryable and attempt < max_retries):
                raise
        except Exception as error:
            retryable = _is_retryable(error)
            if not (retryable and attempt < max_retries):
                raise LLMError(
                    f"{what} failed: {type(error).__name__}: {error}", retryable=retryable
                ) from error
        # Resolved per call so tests can patch `time.sleep`.
        (sleep or time.sleep)(base_delay_s * 2**attempt)
        attempt += 1
