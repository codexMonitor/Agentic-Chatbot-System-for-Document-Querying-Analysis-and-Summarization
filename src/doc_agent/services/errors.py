import os
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MIN_SECRET_LENGTH = 6
MAX_RETRY_WAIT_S = 60.0  # upper bound for one wait, whether it came from the server or the backoff
_JITTER_FRACTION = 0.25
_RETRY_IN_TEXT = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def mask_secrets(text: str) -> str:
    """Replace the value of every configured `*API_KEY` environment variable with `***`."""
    for name, value in os.environ.items():
        if name.upper().endswith("API_KEY") and len(value.strip()) >= _MIN_SECRET_LENGTH:
            text = text.replace(value.strip(), "***")
    return text


class LLMError(Exception):
    """Any failure talking to an LLM or embedding provider (including rate limits).

    `retryable` marks transient failures (rate limits, temporary server errors) that a caller
    may retry; authentication and configuration errors leave it False. `retry_after_s` is the
    delay the provider asked for, if it gave one. The message is masked, so a key echoed back by
    an SDK never reaches logs or the console.
    """

    def __init__(
        self, message: str = "", *, retryable: bool = False, retry_after_s: float | None = None
    ) -> None:
        super().__init__(mask_secrets(message))
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class IndexMismatchError(Exception):
    """The stored vectors were built with a different embedding model than the configured one."""


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def _seconds(value: Any) -> float | None:
    """Parse 5, '5' or '5.5s' (also protobuf-style durations) into seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().removesuffix("s"))
        except ValueError:
            return None
    return None


def retry_after_from(error: Exception) -> float | None:
    """Delay in seconds that a provider error asks for, or None.

    Looks at the structured `details` (Google-style `RetryInfo.retryDelay`), a `Retry-After`
    header on the response, and finally the "Please retry in 5.4s" sentence in the message; when
    several are present the longest wins, because the structured value is often rounded down.
    """
    hints: list[float] = []
    details = getattr(error, "details", None)
    body = details.get("error", details) if isinstance(details, dict) else {}
    for item in (body.get("details") or []) if isinstance(body, dict) else []:
        if isinstance(item, dict) and "retryDelay" in item:
            hints.append(_seconds(item["retryDelay"]))
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is not None and hasattr(headers, "get"):
        hints.append(_seconds(headers.get("retry-after")))
    match = _RETRY_IN_TEXT.search(str(error))
    if match:
        hints.append(float(match.group(1)))
    positive = [hint for hint in hints if hint is not None and hint > 0]
    return max(positive) if positive else None


def _wait_s(
    attempt: int, retry_after_s: float | None, base_delay_s: float, jitter: Callable[[], float]
) -> float:
    if retry_after_s is not None and retry_after_s > 0:
        # The floor keeps a tiny hint (for example 0.05s) from turning into a busy retry loop.
        wait = max(retry_after_s, base_delay_s)
    else:
        wait = base_delay_s * 2**attempt
    # Jitter only ever lengthens the wait, and desynchronises workers that were rejected together.
    return min(wait * (1 + _JITTER_FRACTION * jitter()), MAX_RETRY_WAIT_S)


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int,
    what: str,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] | None = None,
) -> T:
    """Run `func`, retrying rate-limit/server errors; the one retry layer for provider calls.

    At most `max_retries + 1` attempts. A retryable failure waits for the delay the provider asked
    for, else exponential backoff; a little jitter is added to either, and one wait never exceeds
    `MAX_RETRY_WAIT_S`, so the total wait for one call is bounded by `max_retries` times that.
    Every provider exception is re-raised as `LLMError` (flagged `retryable` when it was a
    transient one); an `LLMError` raised by `func` is retried only if it is flagged `retryable`.
    Authentication and other client errors are never retried. The API key is never included.
    """
    attempt = 0
    while True:
        try:
            return func()
        except LLMError as error:
            if not (error.retryable and attempt < max_retries):
                raise
            retry_after_s = error.retry_after_s
        except Exception as error:
            retryable = _is_retryable(error)
            retry_after_s = retry_after_from(error) if retryable else None
            if not (retryable and attempt < max_retries):
                raise LLMError(
                    f"{what} failed: {type(error).__name__}: {error}",
                    retryable=retryable,
                    retry_after_s=retry_after_s,
                ) from error
        # Resolved per call so tests can patch `time.sleep`.
        wait = _wait_s(attempt, retry_after_s, base_delay_s, jitter or random.random)
        (sleep or time.sleep)(wait)
        attempt += 1
