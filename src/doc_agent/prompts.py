ROUTER = """### TASK: route
Classify the user's request. Reply with exactly one word: chat, qa, analysis, or summarize.
- chat: only greetings, thanks, acknowledgments, or questions about what the assistant can do
  or how to use it, such as uploading a document or asking a question.
- qa: a request for a fact, name, number, setting, or detail that may be in the documents.
- analysis: comparing, extracting, or analysing information across the documents.
- summarize: a summary or overview of the documents.
Any request for a fact, name, number, setting, or detail goes to qa, even when it mentions
"the system", "the model", or "the project": uploaded documents may contain the answer.
Questions about document content go to qa, analysis, or summarize, not chat.
When unsure between chat and anything else, choose qa.

Examples:
Request: Good morning!
Intent: chat
Request: How can I add a file?
Intent: chat
Request: Which model is configured for this project?
Intent: qa
Request: Ai là tác giả của báo cáo này?
Intent: qa
Request: Compare the two proposals and list the differences.
Intent: analysis
Request: Extract all deadlines mentioned in the documents.
Intent: analysis
Request: Give me a brief overview of the document.
Intent: summarize
Request: Tóm tắt tài liệu này.
Intent: summarize
---
{question}"""

CHAT = """### TASK: chat
You are a document assistant. You can answer questions about uploaded documents, analyze them,
and summarize them, with citations for document-based answers.
For greetings, thanks, or questions about your capabilities, reply briefly and helpfully.
Do not answer general-knowledge questions or questions about document contents from your own
knowledge. If asked something outside the uploaded documents or your capabilities, say that you
only help with uploaded documents.
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
Write the summary in the same language as the text.
---
{chunk}"""

SUMMARIZE_REDUCE = """### TASK: summarize_reduce
Combine these partial summaries, which are in document order, into one coherent summary
of the text they cover. Remove repetition and keep the key facts.
Write the summary in the same language as the partial summaries.
---
{partials}"""

REWRITE = """### TASK: rewrite
Rewrite the new question as one standalone question that can be understood without the conversation.
Use the conversation only to resolve references such as "it", "they" or "that".
Keep the language of the new question and change nothing else.
If the question is already standalone, repeat it unchanged.
Reply with the question only.
---
Conversation:
{conversation}

New question: {question}"""
