import pytest

from doc_agent.nodes._utils import check_citations


def test_check_citations_returns_empty_when_there_are_no_citations():
    assert check_citations("No citations here.", n_sources=2) == []


def test_check_citations_accepts_valid_citations():
    assert check_citations("Facts [1] and [2].", n_sources=2) == []


def test_check_citations_returns_out_of_range_numbers():
    assert check_citations("Invalid [0] and [3].", n_sources=2) == [0, 3]


def test_check_citations_preserves_repeated_invalid_citations():
    assert check_citations("Unsupported [3], repeated [3].", n_sources=2) == [3, 3]


def test_check_citations_rejects_negative_source_count():
    with pytest.raises(ValueError, match="must not be negative"):
        check_citations("No citations.", n_sources=-1)
