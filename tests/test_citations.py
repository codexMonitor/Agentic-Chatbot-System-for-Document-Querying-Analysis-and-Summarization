import pytest

from doc_agent.nodes._utils import check_citations, source_refs
from doc_agent.services.base import Chunk


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


def test_table_source_reference_includes_sheet_and_row_range():
    chunk = Chunk(
        'Citation: report.xlsx, Sheet "Q3", rows 12-40\nSheet: "Q3"\n| Revenue |',
        "report.xlsx",
        3,
    )

    assert source_refs([chunk]) == [
        {"n": 1, "source": 'report.xlsx, Sheet "Q3", rows 12-40', "position": 3}
    ]
