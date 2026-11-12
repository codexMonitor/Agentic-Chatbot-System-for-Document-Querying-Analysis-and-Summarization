import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from doc_agent.services.ingestion import LOADERS, ingest_file, load_text
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
    assert set(LOADERS) == {".txt", ".md", ".pdf", ".docx"}


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
