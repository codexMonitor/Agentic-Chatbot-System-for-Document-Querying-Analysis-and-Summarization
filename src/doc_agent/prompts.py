ROUTER = """### TASK: route
Classify the user's request. Reply with exactly one word: qa, analysis, or summarize.
- qa: a specific question answerable from the documents.
- analysis: comparing, extracting, or analysing information across the documents.
- summarize: a summary or overview of the documents.
---
{question}"""

QA = """### TASK: qa
Answer the question using only the numbered context below.
Cite every fact with the number of the chunk it came from, like [1] or [2].
If the answer is not in the context, reply that it is not in the documents.
Answer in the language of the question.
---
Context:
{context}

Question: {question}"""

ANALYSIS = """### TASK: analysis
Analyse the document excerpts to fulfil the request.
Base your analysis only on the excerpts and cite chunk numbers like [1].
---
Document excerpts:
{context}

Request: {question}"""

SUMMARIZE_CHUNK = """### TASK: summarize_chunk
Summarize the following text in one or two sentences.
---
{chunk}"""

SUMMARIZE_REDUCE = """### TASK: summarize_reduce
Combine these partial summaries into one coherent summary of the whole document.
---
{partials}"""
