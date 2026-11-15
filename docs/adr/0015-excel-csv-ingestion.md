# ADR-0015: Read Excel and CSV tables as cited text chunks

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

Internal financial reports are often stored in Excel workbooks or CSV files. The existing ingestion
pipeline handles document text but has no table-aware representation or per-file table limits.

## Decision

Read `.xlsx` files with `openpyxl` in cached-value mode and `.csv` files as UTF-8 (including a UTF-8
BOM). Render tables as Markdown-style text and split them into row groups that repeat the header and
Excel sheet name. Include the source row range in each chunk so the existing citation path can show
the table location. Skip hidden worksheets by default, with an environment setting to include them.
Reject legacy `.xls` and macro-enabled files. Limit uploads to the configured size and each table
file to 20,000 used-range rows and 200,000 used-range cells by default. The cell limit counts blank
cells inside the used range as well as populated cells to bound memory use.

## Consequences

CLI, API and UI uploads use one loader. Formula results are only as current as the workbook's cached
values; missing cached results stay blank and produce a warning. The language model still reasons
over retrieved text chunks, so it is not an exact calculation engine for large tables.
