import csv
import math
import posixpath
import re
import warnings
from collections.abc import Callable
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from doc_agent.config import (
    DEFAULT_MAX_TABLE_CELLS,
    DEFAULT_MAX_TABLE_ROWS,
    DEFAULT_MAX_UPLOAD_MB,
)
from doc_agent.services.base import VectorStore

MACRO_SUFFIXES = {".xlsm", ".xlsb", ".xltm", ".xlam"}
_CITATION_PREFIX = "Citation: "
_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_OFFICE_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL_REF_RE = re.compile(r"\$?([A-Z]+)\$?(\d+)")


def _column_number(reference: str) -> int:
    match = _CELL_REF_RE.fullmatch(reference)
    if match is None:
        return 0
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result


def _dimension_bounds(reference: str) -> tuple[int, int]:
    end = reference.rsplit(":", 1)[-1]
    match = _CELL_REF_RE.fullmatch(end)
    if match is None:
        return 0, 0
    return int(match.group(2)), _column_number(end)


def _preflight_xlsx(
    path: Path, include_hidden_sheets: bool, max_rows: int, max_cells: int
) -> None:
    try:
        with ZipFile(path) as archive:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
            relationships_by_id = {
                item.attrib["Id"]: item.attrib
                for item in relationships.findall(f"{_PACKAGE_REL_NS}Relationship")
            }
            total_rows = 0
            total_cells = 0
            for sheet in workbook.iter(f"{_MAIN_NS}sheet"):
                if not include_hidden_sheets and sheet.attrib.get("state", "visible") != "visible":
                    continue
                relationship = relationships_by_id[sheet.attrib[f"{_OFFICE_REL_NS}id"]]
                if not relationship["Type"].endswith("/worksheet"):
                    continue
                target = relationship["Target"]
                worksheet_path = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath(posixpath.join("xl", target))
                )
                sheet_rows = 0
                sheet_columns = 0
                with archive.open(worksheet_path) as source:
                    for _, element in ElementTree.iterparse(source, events=("end",)):
                        if element.tag == f"{_MAIN_NS}dimension":
                            rows, columns = _dimension_bounds(element.attrib["ref"])
                            sheet_rows = max(sheet_rows, rows)
                            sheet_columns = max(sheet_columns, columns)
                        elif element.tag == f"{_MAIN_NS}row":
                            sheet_rows = max(
                                sheet_rows, int(element.attrib.get("r", "0"))
                            )
                        elif element.tag == f"{_MAIN_NS}c":
                            row, column = _dimension_bounds(element.attrib["r"])
                            sheet_rows = max(sheet_rows, row)
                            sheet_columns = max(sheet_columns, column)
                            element.clear()
                        if total_rows + sheet_rows > max_rows:
                            raise ValueError(
                                f"{path.name} exceeds DOC_AGENT_MAX_TABLE_ROWS ({max_rows})"
                            )
                        if total_cells + sheet_rows * sheet_columns > max_cells:
                            raise ValueError(
                                f"{path.name} exceeds DOC_AGENT_MAX_TABLE_CELLS ({max_cells})"
                            )
                        if element.tag == f"{_MAIN_NS}row":
                            element.clear()
                total_rows += sheet_rows
                total_cells += sheet_rows * sheet_columns
    except ValueError:
        raise
    except (BadZipFile, KeyError, OSError, ElementTree.ParseError) as error:
        raise ValueError(f"cannot read Excel workbook {path.name}: {error}") from error


def _load_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8 text") from error


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise ValueError(f"cannot read PDF {path.name}: {error}") from error


def _load_docx(path: Path) -> str:
    from docx import Document

    try:
        document = Document(str(path))
    except Exception as error:
        raise ValueError(f"cannot read DOCX {path.name}: {error}") from error
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _load_csv(path: Path) -> str:
    return "\n\n".join(
        _load_csv_chunks(path, 500, DEFAULT_MAX_TABLE_ROWS, DEFAULT_MAX_TABLE_CELLS)
    )


def _load_xlsx(path: Path) -> str:
    return "\n\n".join(
        _load_xlsx_chunks(path, 500, False, DEFAULT_MAX_TABLE_ROWS, DEFAULT_MAX_TABLE_CELLS)
    )


LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_plain,
    ".md": _load_plain,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".xlsx": _load_xlsx,
}


def unsupported_type_message(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".xls":
        return "legacy Excel .xls files are not supported; save the file as .xlsx"
    if suffix in MACRO_SUFFIXES:
        return "macro-enabled Excel files are not supported; remove the macros and save as .xlsx"
    supported = ", ".join(sorted(LOADERS))
    return f"unsupported file type '{suffix}' (supported: {supported})"


def _validate_file(path: Path, max_upload_mb: int) -> Callable[[Path], str]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(unsupported_type_message(path.suffix))
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    if max_upload_mb < 1:
        raise ValueError("max_upload_mb must be at least 1")
    if path.stat().st_size > max_upload_mb * 1024 * 1024:
        raise ValueError(f"file exceeds the {max_upload_mb} MB upload limit")
    return loader


def load_text(
    path: Path,
    *,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_table_cells: int = DEFAULT_MAX_TABLE_CELLS,
    include_hidden_sheets: bool = False,
) -> str:
    if max_table_rows < 1 or max_table_cells < 1:
        raise ValueError("max_table_rows and max_table_cells must be positive")
    loader = _validate_file(path, max_upload_mb)
    if path.suffix.lower() == ".csv":
        text = "\n\n".join(
            _load_csv_chunks(path, 500, max_table_rows, max_table_cells)
        )
    elif path.suffix.lower() == ".xlsx":
        text = "\n\n".join(
            _load_xlsx_chunks(
                path,
                500,
                include_hidden_sheets,
                max_table_rows,
                max_table_cells,
            )
        )
    else:
        text = loader(path)
    if not text.strip():
        raise ValueError(f"no extractable text in {path.name} (scanned documents need OCR)")
    return text


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
    text = text.strip()
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (int, float)):
        return format(value, ",")
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _markdown_row(values: list[Any]) -> str:
    rendered = [_cell_text(value).replace("|", r"\|") for value in values]
    return "| " + " | ".join(rendered) + " |"


def _table_chunk(
    filename: str,
    sheet_name: str | None,
    header: list[Any],
    rows: list[tuple[int, list[Any]]],
    row_range: tuple[int, int] | None = None,
) -> str:
    start, end = row_range or (rows[0][0], rows[-1][0])
    sheet = f'Sheet "{sheet_name}", ' if sheet_name is not None else ""
    label = f"{filename}, {sheet}rows {start}-{end}"
    lines = [f"{_CITATION_PREFIX}{label}"]
    if sheet_name is not None:
        lines.append(f'Sheet: "{sheet_name}"')
    lines.extend((_markdown_row(header), _markdown_row(["---"] * len(header))))
    lines.extend(_markdown_row(values) for _, values in rows)
    return "\n".join(lines)


def _make_row_chunks(
    filename: str,
    sheet_name: str | None,
    header_row: tuple[int, list[Any]],
    data_rows: list[tuple[int, list[Any]]],
    chunk_size: int,
) -> list[str]:
    header = header_row[1]
    if not data_rows:
        return [
            _table_chunk(
                filename,
                sheet_name,
                header,
                [],
                row_range=(header_row[0], header_row[0]),
            )
        ]
    groups: list[str] = []
    current: list[tuple[int, list[Any]]] = []
    for row in data_rows:
        proposed = current + [row]
        rendered = _table_chunk(filename, sheet_name, header, proposed)
        if current and len(rendered) > chunk_size:
            groups.append(_table_chunk(filename, sheet_name, header, current))
            current = [row]
        else:
            current = proposed
    if current:
        groups.append(_table_chunk(filename, sheet_name, header, current))
    return groups


def _load_csv_chunks(
    path: Path, chunk_size: int, max_rows: int, max_cells: int
) -> list[str]:
    rows = []
    max_width = 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if row_number > max_rows:
                    raise ValueError(
                        f"{path.name} exceeds DOC_AGENT_MAX_TABLE_ROWS ({max_rows})"
                    )
                rows.append(row)
                max_width = max(max_width, len(row))
                if row_number * max_width > max_cells:
                    raise ValueError(
                        f"{path.name} exceeds DOC_AGENT_MAX_TABLE_CELLS ({max_cells})"
                    )
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8 CSV") from error
    except csv.Error as error:
        raise ValueError(f"cannot read CSV {path.name}: {error}") from error
    nonempty_rows = [(number, row) for number, row in enumerate(rows, start=1) if any(
        not _is_empty(value) for value in row
    )]
    if not nonempty_rows:
        return []

    header_number, header_values = nonempty_rows[0]
    width = max(len(row) for _, row in nonempty_rows)
    expanded = [
        (number, row + [""] * (width - len(row))) for number, row in nonempty_rows
    ]
    used_columns = [
        index
        for index in range(width)
        if any(not _is_empty(row[index]) for _, row in expanded)
    ]
    compact = [(number, [row[index] for index in used_columns]) for number, row in expanded]
    header = next(row for number, row in compact if number == header_number)
    data_rows = [(number, row) for number, row in compact if number != header_number]
    return _make_row_chunks(
        path.name,
        None,
        (header_number, header),
        data_rows,
        chunk_size,
    )


def _load_xlsx_chunks(
    path: Path,
    chunk_size: int,
    include_hidden_sheets: bool,
    max_rows: int,
    max_cells: int,
) -> list[str]:
    from openpyxl import load_workbook

    _preflight_xlsx(path, include_hidden_sheets, max_rows, max_cells)
    try:
        cached_book = load_workbook(path, data_only=True)
    except Exception as error:
        raise ValueError(f"cannot read Excel workbook {path.name}: {error}") from error
    try:
        formula_book = load_workbook(path, data_only=False)
    except Exception as error:
        cached_book.close()
        raise ValueError(f"cannot read Excel workbook {path.name}: {error}") from error

    chunks: list[str] = []
    try:
        for cached_sheet, formula_sheet in zip(
            cached_book.worksheets, formula_book.worksheets, strict=True
        ):
            if not include_hidden_sheets and cached_sheet.sheet_state != "visible":
                continue
            sheet_rows: list[tuple[int, list[Any]]] = []
            missing_cache_count = 0
            missing_cache_samples = []
            for row_number, formula_row in enumerate(formula_sheet.iter_rows(), start=1):
                cached_row = [
                    cached_sheet.cell(row_number, column_number)
                    for column_number in range(1, len(formula_row) + 1)
                ]
                formula_values = [cell.value for cell in formula_row]
                for formula_cell, cached_cell in zip(formula_row, cached_row, strict=True):
                    if formula_cell.data_type == "f" and cached_cell.value is None:
                        missing_cache_count += 1
                        if len(missing_cache_samples) < 5:
                            missing_cache_samples.append(formula_cell.coordinate)
                if any(not _is_empty(value) for value in formula_values):
                    sheet_rows.append(
                        (row_number, [cell.value for cell in cached_row])
                    )
            if missing_cache_count:
                samples = ", ".join(missing_cache_samples)
                warnings.warn(
                    f"{path.name}, sheet {cached_sheet.title!r}, has "
                    f"{missing_cache_count} formula(s) with no cached value"
                    f" (for example: {samples})",
                    stacklevel=2,
                )

            if not sheet_rows:
                continue
            width = max(len(values) for _, values in sheet_rows)
            expanded = [
                (number, values + [None] * (width - len(values)))
                for number, values in sheet_rows
            ]
            used_columns = [
                index
                for index in range(width)
                if any(not _is_empty(values[index]) for _, values in expanded)
            ]
            compact = [
                (number, [values[index] for index in used_columns])
                for number, values in expanded
            ]
            header_row = compact[0]
            data_rows = compact[1:]
            chunks.extend(
                _make_row_chunks(
                    path.name,
                    cached_sheet.title,
                    header_row,
                    data_rows,
                    chunk_size,
                )
            )
    finally:
        cached_book.close()
        formula_book.close()
    return chunks


def load_file_chunks(
    path: Path,
    chunk_size: int,
    overlap: int,
    *,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_table_cells: int = DEFAULT_MAX_TABLE_CELLS,
    include_hidden_sheets: bool = False,
) -> list[str]:
    if max_table_rows < 1 or max_table_cells < 1:
        raise ValueError("max_table_rows and max_table_cells must be positive")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
    loader = _validate_file(path, max_upload_mb)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        chunks = _load_csv_chunks(path, chunk_size, max_table_rows, max_table_cells)
    elif suffix == ".xlsx":
        chunks = _load_xlsx_chunks(
            path,
            chunk_size,
            include_hidden_sheets,
            max_table_rows,
            max_table_cells,
        )
    else:
        chunks = split_text(loader(path), chunk_size, overlap)
    if not chunks:
        raise ValueError(f"no extractable text in {path.name} (scanned documents need OCR)")
    return chunks


def ingest_file(
    path: Path,
    store: VectorStore,
    chunk_size: int,
    overlap: int,
    *,
    source: str | None = None,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_table_cells: int = DEFAULT_MAX_TABLE_CELLS,
    include_hidden_sheets: bool = False,
) -> int:
    chunks = load_file_chunks(
        path,
        chunk_size,
        overlap,
        max_upload_mb=max_upload_mb,
        max_table_rows=max_table_rows,
        max_table_cells=max_table_cells,
        include_hidden_sheets=include_hidden_sheets,
    )
    store.add(source or path.name, chunks)  # the store assigns positions
    return len(chunks)
