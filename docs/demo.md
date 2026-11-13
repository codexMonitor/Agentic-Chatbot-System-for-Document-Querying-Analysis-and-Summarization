# Local UI demo

## Manual checklist

- [ ] Start the UI in HTTP mode with the FastAPI server available, or in embedded mode.
- [ ] Confirm the privacy note recommends public documents only.
- [ ] Upload a supported public TXT, MD, PDF, or DOCX file. Include a filename with spaces or Vietnamese characters.
- [ ] Confirm the document appears in the sidebar and can be deleted.
- [ ] Ask a factual question and confirm sources appear below the answer in an expander.
- [ ] Ask for a summary and a structured analysis.
- [ ] Start a new conversation, then reopen the earlier conversation from the sidebar.
- [ ] Confirm the earlier messages are restored; delete a conversation and verify it disappears.

## Five-minute demo script

1. **0:00–0:45 — Start and explain privacy.** Open the local app and point out that documents and questions are sent to the configured LLM provider, so this demo uses public documents.
2. **0:45–1:30 — Upload.** Upload a public document with a filename containing spaces or Vietnamese characters. Show it in the document list.
3. **1:30–2:15 — Ask and show sources.** Ask a question answered by the document, then open the Sources expander and identify the source file and chunk.
4. **2:15–3:00 — Summarize.** Ask “Summarize the document” and note that a long summary can take several minutes.
5. **3:00–3:45 — Analyze.** Ask for a comparison or structured analysis and show the result.
6. **3:45–5:00 — Reopen history.** Start another conversation, return to the first conversation from the sidebar, show that its messages were restored, and point out the title/date and delete action.
