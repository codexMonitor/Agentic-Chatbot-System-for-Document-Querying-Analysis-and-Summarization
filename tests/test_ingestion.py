import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest
from docx import Document
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from doc_agent.services.ingestion import LOADERS, ingest_file, load_file_chunks, load_text
from doc_agent.services.vectorstore import InMemoryVectorStore


def make_pdf(path, text):
    """Write a one-page PDF whose text layer is `text` (empty text gives a blank page)."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def test_registry_covers_expected_suffixes():
    assert set(LOADERS) == {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}


def test_load_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("Xin chào thế giới", encoding="utf-8")
    (tmp_path / "b.MD").write_text("# Title\nbody", encoding="utf-8")
    assert load_text(tmp_path / "a.txt") == "Xin chào thế giới"
    assert "body" in load_text(tmp_path / "b.MD")


def test_load_docx(tmp_path):
    path = tmp_path / "doc.docx"
    document = Document()
    document.add_paragraph("Tài liệu về Scrum.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "cell A"
    table.rows[0].cells[1].text = "cell B"
    document.save(str(path))
    text = load_text(path)
    assert "Tài liệu về Scrum." in text
    assert "cell A | cell B" in text


def test_load_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    make_pdf(path, "Hello PDF world")
    assert "Hello PDF world" in load_text(path)


def test_ingest_docx_end_to_end(tmp_path):
    path = tmp_path / "doc.docx"
    document = Document()
    document.add_paragraph("Scrum uses sprints.")
    document.save(str(path))
    store = InMemoryVectorStore()
    assert ingest_file(path, store, 500, 50) == 1
    assert store.all_chunks()[0].source == "doc.docx"


def test_unsupported_suffix(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_text(path)


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="file not found"):
        load_text(tmp_path / "nope.txt")


@pytest.mark.parametrize("suffix", [".txt", ".docx", ".pdf"])
def test_empty_text(tmp_path, suffix):
    path = tmp_path / f"empty{suffix}"
    if suffix == ".txt":
        path.write_text("  \n ", encoding="utf-8")
    elif suffix == ".docx":
        Document().save(str(path))
    else:
        make_pdf(path, "")
    with pytest.raises(ValueError, match="no extractable text"):
        load_text(path)


def test_unreadable_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError, match="cannot read PDF"):
        load_text(path)


def test_unreadable_docx(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a docx")
    with pytest.raises(ValueError, match="cannot read DOCX"):
        load_text(path)


def test_non_utf8_text(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(ValueError, match="UTF-8"):
        load_text(path)


def _inject_cached_formula_value(path: Path, coordinate: str, value: str) -> None:
    source = path.with_suffix(".source.xlsx")
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(source, "w") as updated:
        for item in original.infolist():
            contents = original.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                root = ElementTree.fromstring(contents)
                cell = next(
                    item
                    for item in root.iter(f"{namespace}c")
                    if item.attrib.get("r") == coordinate
                )
                cached_value = cell.find(f"{namespace}v")
                if cached_value is None:
                    cached_value = ElementTree.SubElement(cell, f"{namespace}v")
                cached_value.text = value
                contents = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            updated.writestr(item, contents)
    source.replace(path)


def _make_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Q3"
    sheet.append(["Region", "Revenue", "As of", ""])
    sheet.merge_cells("A2:A3")
    sheet["A2"] = "North"
    sheet["B2"] = 1200
    sheet["C2"] = datetime(2026, 9, 30)
    sheet["B3"] = 850
    sheet["C3"] = datetime(2026, 9, 30)
    sheet["A4"] = "South"
    sheet["B4"] = "=SUM(B2:B3)"
    sheet["A5"] = "West"
    sheet["B5"] = "=B2*2"
    for row in range(7, 13):
        sheet.cell(row, 1, f"Region {row}")
        sheet.cell(row, 2, row * 100)

    second = workbook.create_sheet("Summary")
    second.append(["Metric", "Value"])
    second.append(["Regions", 3])
    hidden = workbook.create_sheet("Hidden")
    hidden.append(["Private", "Value"])
    hidden.append(["Secret", 7])
    hidden.sheet_state = "hidden"
    workbook.save(path)
    _inject_cached_formula_value(path, "B5", "2400")


def test_xlsx_tables_repeat_sheet_header_and_citation_per_chunk(tmp_path):
    path = tmp_path / "report.xlsx"
    _make_workbook(path)

    with pytest.warns(UserWarning, match="B4"):
        chunks = load_file_chunks(path, 90, 0)

    q3_chunks = [chunk for chunk in chunks if 'Sheet "Q3"' in chunk]
    assert len(q3_chunks) > 2
    for chunk in q3_chunks:
        assert "Citation: report.xlsx, Sheet \"Q3\", rows " in chunk
        assert 'Sheet: "Q3"' in chunk
        assert "| Region | Revenue | As of |" in chunk
    assert "2,400" in "\n".join(q3_chunks)
    assert "=B2*2" not in "\n".join(q3_chunks)
    assert "2026-09-30" in "\n".join(q3_chunks)
    assert "1,200" in "\n".join(q3_chunks)
    assert "| North | 1,200 |" in "\n".join(q3_chunks)
    assert "|  | 850 |" in "\n".join(q3_chunks)
    assert "rows 6-6" not in "\n".join(q3_chunks)
    assert "Secret" not in "\n".join(chunks)


def test_xlsx_can_include_hidden_sheets_and_preserves_file_source(tmp_path):
    path = tmp_path / "report.xlsx"
    _make_workbook(path)
    store = InMemoryVectorStore()
    with pytest.warns(UserWarning):
        count = ingest_file(path, store, 500, 50, include_hidden_sheets=True)

    assert count == len(store.all_chunks())
    assert all(chunk.source == "report.xlsx" for chunk in store.all_chunks())
    assert any("Secret" in chunk.text for chunk in store.all_chunks())


def test_xlsx_limits_have_clear_messages(tmp_path):
    path = tmp_path / "report.xlsx"
    _make_workbook(path)
    with pytest.raises(ValueError, match="DOC_AGENT_MAX_TABLE_ROWS"):
        load_file_chunks(path, 500, 50, max_table_rows=2)
    with pytest.raises(ValueError, match="DOC_AGENT_MAX_TABLE_CELLS"):
        load_file_chunks(path, 500, 50, max_table_cells=2)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("legacy.xls", "legacy Excel .xls files are not supported"),
        ("macro.xlsm", "macro-enabled Excel files are not supported"),
    ],
)
def test_xls_and_macro_workbooks_are_rejected(tmp_path, filename, message):
    path = tmp_path / filename
    path.write_bytes(b"not opened")
    with pytest.raises(ValueError, match=message):
        load_file_chunks(path, 500, 50)


@pytest.mark.parametrize("bom", [False, True])
def test_csv_handles_quotes_commas_vietnamese_and_bom(tmp_path, bom):
    path = tmp_path / "report.csv"
    contents = 'Tên,Ghi chú\nLan,"Báo cáo, ""tốt"""\nMinh,"Xin chào"\n'
    encoded = contents.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + encoded)

    chunks = load_file_chunks(path, 500, 50)

    assert len(chunks) == 1
    assert "Citation: report.csv, rows 2-3" in chunks[0]
    assert "| Tên | Ghi chú |" in chunks[0]
    assert '| Lan | Báo cáo, "tốt" |' in chunks[0]
    assert "Xin chào" in chunks[0]
    assert "\ufeff" not in chunks[0]


def test_csv_table_limits_and_utf8_errors_are_reported(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("A,B\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="DOC_AGENT_MAX_TABLE_ROWS"):
        load_file_chunks(path, 500, 50, max_table_rows=2)
    with pytest.raises(ValueError, match="DOC_AGENT_MAX_TABLE_CELLS"):
        load_file_chunks(path, 500, 50, max_table_cells=2)

    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8 CSV"):
        load_file_chunks(path, 500, 50)


def test_upload_size_limit_is_shared_by_ingestion(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(ValueError, match="1 MB upload limit"):
        load_file_chunks(path, 500, 50, max_upload_mb=1)
