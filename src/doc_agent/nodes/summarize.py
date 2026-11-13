import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TypeVar

from doc_agent.nodes._utils import guard_llm_errors, reply
from doc_agent.prompts import SUMMARIZE_CHUNK, SUMMARIZE_REDUCE
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _parallel_map(func: Callable[[T], R], items: list[T], workers: int) -> list[R]:
    """Like `map` on a thread pool: results keep input order and the first error stops the rest."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_context().run, func, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def make_summarize(
    llm: LLMService,
    store: VectorStore,
    workers: int = 2,
    group_size: int = 8,
):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")

    # `llm` already retries rate limits itself (see `call_with_retry`); a second layer here would
    # multiply the attempts, so a worker's failure after those retries stops the summary.
    def merge(group: list[str]) -> str:
        if len(group) == 1:
            return group[0]
        bullets = "\n".join(f"- {summary}" for summary in group)
        return llm.invoke(SUMMARIZE_REDUCE.format(partials=bullets))

    def run(state: AgentState) -> dict:
        chunks = store.all_chunks()
        if not chunks:
            return reply("No documents have been ingested yet.", context=[], sources=[])
        summaries = _parallel_map(
            lambda chunk: llm.invoke(SUMMARIZE_CHUNK.format(chunk=chunk.text)), chunks, workers
        )
        while len(summaries) > 1:
            groups = [summaries[i : i + group_size] for i in range(0, len(summaries), group_size)]
            logger.info("summarize: merging %d summaries into %d", len(summaries), len(groups))
            summaries = _parallel_map(merge, groups, workers)
        # Summaries are not cited per chunk; empty sources also clear stale ones from earlier turns.
        return reply(summaries[0], context=chunks, sources=[])

    return guard_llm_errors("summarize", run)
