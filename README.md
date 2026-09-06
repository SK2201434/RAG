# Enterprise RAG Assistant

A conversational Retrieval-Augmented Generation (RAG) application built with Python, Docling, FAISS, BM25, Cross-Encoder reranking, and Ollama.

The project implements an end-to-end RAG pipeline for asking questions about PDF documents through an interactive conversational CLI.

The main goal of this project is to understand and implement the complete RAG architecture, from document ingestion and indexing to retrieval, reranking, context construction, LLM generation, conversation history, and deterministic citations.

The current version is an **MVP** focused on building and understanding a complete working RAG application.

---

# Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. What is RAG?](#2-what-is-rag)
- [3. Why RAG?](#3-why-rag)
- [4. Project Goals](#4-project-goals)
- [5. Architecture](#5-architecture)
- [6. End-to-End RAG Flow](#6-end-to-end-rag-flow)
- [7. Project Structure](#7-project-structure)
- [8. Technology Stack](#8-technology-stack)
- [9. Document Ingestion](#9-document-ingestion)
- [10. Hybrid Chunking](#10-hybrid-chunking)
- [11. Document Metadata](#11-document-metadata)
- [12. Embeddings](#12-embeddings)
- [13. FAISS Dense Retrieval](#13-faiss-dense-retrieval)
- [14. BM25 Sparse Retrieval](#14-bm25-sparse-retrieval)
- [15. Hybrid Retrieval](#15-hybrid-retrieval)
- [16. Reciprocal Rank Fusion](#16-reciprocal-rank-fusion)
- [17. Cross-Encoder Reranking](#17-cross-encoder-reranking)
- [18. Context Expansion](#18-context-expansion)
- [19. Context Builder](#19-context-builder)
- [20. Conversation History](#20-conversation-history)
- [21. Prompt Builder](#21-prompt-builder)
- [22. Local LLM Generation](#22-local-llm-generation)
- [23. Citation Builder](#23-citation-builder)
- [24. RAG Pipeline](#24-rag-pipeline)
- [25. Conversational CLI](#25-conversational-cli)
- [26. Configuration](#26-configuration)
- [27. End-to-End Example](#27-end-to-end-example)
- [28. Testing](#28-testing)
- [29. Current MVP](#29-current-mvp)
- [30. Current Limitations](#30-current-limitations)
- [31. Future Improvements](#31-future-improvements)
- [32. Important RAG Concepts](#32-important-rag-concepts)
- [33. RAG Revision Cheat Sheet](#33-rag-revision-cheat-sheet)
- [34. Learning Roadmap](#34-learning-roadmap)
- [35. Version](#35-version)

---

# 1. Project Overview

The Enterprise RAG Assistant is a conversational document-question-answering system.

The user provides a question through a CLI.

The system then:

1. Searches the indexed document using semantic retrieval.
2. Searches the document using keyword-based retrieval.
3. Combines the retrieval results.
4. Reranks the most relevant candidates.
5. Expands the context using neighboring chunks.
6. Builds a context within a token budget.
7. Adds conversation history.
8. Builds the LLM prompt.
9. Sends the prompt to a local LLM through Ollama.
10. Generates the answer.
11. Generates deterministic citations from document metadata.
12. Saves the conversation locally.

The core idea is:

```text
User Question
      ↓
Retrieve Evidence
      ↓
Build Context
      ↓
Generate Answer
      ↓
Attach Citations
```

---

# 2. What is RAG?

RAG stands for:

**Retrieval-Augmented Generation**

A normal LLM application can be represented as:

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

The LLM relies primarily on knowledge learned during training.

A RAG application introduces a retrieval layer:

```text
User Question
      ↓
   Retrieval
      ↓
Relevant Documents
      ↓
    Context
      ↓
     LLM
      ↓
   Answer
```

Therefore:

```text
RAG = Retrieval + Augmentation + Generation
```

## Retrieval

Find relevant information from an external knowledge source.

## Augmentation

Add the retrieved information to the LLM's input.

## Generation

Use the LLM to generate an answer using the retrieved information.

---

# 3. Why RAG?

RAG is useful when an application needs to answer questions using information that is:

- private
- domain-specific
- frequently changing
- too large to put directly into a prompt
- contained in enterprise documents
- required to have source references

Typical use cases include:

```text
Company policies
Employee handbooks
Legal documents
Technical documentation
Financial reports
Research papers
Product documentation
Internal knowledge bases
```

Instead of retraining an LLM every time a document changes, the document can be indexed and retrieved at query time.

---

# 4. Project Goals

The main goals of this project are:

- Understand the complete RAG architecture.
- Build each RAG component independently.
- Integrate the components into one pipeline.
- Support conversational questions.
- Use both semantic and keyword retrieval.
- Improve retrieval quality using reranking.
- Preserve document structure and metadata.
- Provide deterministic citations.
- Prevent obvious hallucinations when information is unavailable.
- Run the LLM locally using Ollama.
- Provide a simple interactive CLI.

This project currently focuses on building a working **MVP** rather than a production deployment.

---

# 5. Architecture

The current architecture is:

```text
                         USER
                          │
                          ▼
                Conversational CLI
                          │
                          ▼
                    RAGPipeline
                          │
                          ▼
                    User Question
                          │
                          ▼
              ┌───────────────────────┐
              │   Hybrid Retrieval    │
              └───────────┬───────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
          FAISS                      BM25
       Dense Search              Sparse Search
             │                         │
             └────────────┬────────────┘
                          ▼
                   RRF Fusion
                          │
                          ▼
                Candidate Documents
                          │
                          ▼
                Cross-Encoder
                  Reranking
                          │
                          ▼
                 Context Expansion
                          │
                          ▼
                  Context Builder
                          │
                          ▼
                 Prompt Builder
                          │
                          ▼
                     Ollama
                          │
                          ▼
                    Gemma 3 4B
                          │
                          ▼
                 Generated Answer
                          │
                          ▼
                 Citation Builder
                          │
                          ▼
                Answer + Citations
                          │
                          ▼
                Conversation Storage
```

---

# 6. End-to-End RAG Flow

The application can be divided into two major parts.

## A. Indexing

```text
PDF
 ↓
Docling
 ↓
Structured Document
 ↓
Hybrid Chunking
 ↓
Metadata Extraction
 ↓
Embedding Generation
 ↓
FAISS Index
```

## B. Querying

```text
User Question
 ↓
FAISS Retrieval
 ↓
BM25 Retrieval
 ↓
RRF
 ↓
Cross-Encoder Reranking
 ↓
Context Expansion
 ↓
Context Builder
 ↓
Conversation History
 ↓
Prompt Builder
 ↓
Ollama / Gemma
 ↓
Generated Answer
 ↓
Citation Builder
 ↓
Final Response
 ↓
Save Conversation
```

---

# 7. Project Structure

The project is organized into separate modules so each part of the RAG system has a clear responsibility.

```text
enterprise-rag/
│
├── app/
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── app.py
│   │
│   ├── chunking/
│   │   └── docling_chunker.py
│   │
│   ├── conversation/
│   │   └── history.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── generation/
│   │   ├── citation_builder.py
│   │   ├── context_builder.py
│   │   ├── llm.py
│   │   ├── prompt_builder.py
│   │   └── token_counter.py
│   │
│   ├── indexing/
│   │   └── indexer.py
│   │
│   ├── ingestion/
│   │   └── docling_parser.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── context_expander.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   │
│   ├── vectorstore/
│   │   └── faiss_store.py
│   │
│   └── config.py
│
├── data/
│   ├── company-policy.pdf
│   ├── faiss_index/
│   └── conversations/
│
├── tests/
│   ├── test_bm25.py
│   ├── test_citation_builder.py
│   ├── test_context_builder.py
│   ├── test_context_expansion.py
│   ├── test_conversation_history.py
│   ├── test_docling.py
│   ├── test_hybrid.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_prompt_builder.py
│   ├── test_rag_generation.py
│   └── test_reranker.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 8. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | Docling |
| Chunking | Docling HybridChunker |
| Embeddings | Sentence Transformers |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | FAISS |
| Sparse Retrieval | BM25 |
| Hybrid Fusion | Reciprocal Rank Fusion |
| Reranking | Sentence Transformers Cross-Encoder |
| Reranker Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM Runtime | Ollama |
| LLM | `gemma3:4b` |
| LLM Framework | LangChain / LangChain Ollama |
| Conversation Storage | Local JSON |
| Interface | Interactive CLI |

---

# 9. Document Ingestion

The ingestion stage converts the PDF into a structured representation.

The project uses Docling.

Relevant module:

```text
app/ingestion/docling_parser.py
```

Flow:

```text
PDF
 ↓
Docling
 ↓
Structured Document
```

The purpose is to preserve useful document structure instead of treating the PDF as a simple block of plain text.

Information such as headings, document items, and page provenance can be preserved.

---

# 10. Hybrid Chunking

The project uses Docling's `HybridChunker`.

Chunking divides a large document into smaller units called chunks.

Example:

```text
Large PDF
   │
   ├── Chunk 1
   ├── Chunk 2
   ├── Chunk 3
   ├── Chunk 4
   └── Chunk 5
```

Chunking is important because sending an entire large document to the LLM for every question is inefficient and can exceed the model's context window.

The project uses document structure while creating chunks instead of blindly splitting text by character count.

Relevant module:

```text
app/chunking/docling_chunker.py
```

---

# 11. Document Metadata

Each chunk receives metadata.

Example:

```python
{
    "document_id": "company-policy",
    "chunk_id": "company-policy:chunk_000008",
    "heading": "III. Duties of Independent Directors",
    "page_numbers": [4]
}
```

Metadata allows the system to track:

```text
Document
Section
Page
Chunk
```

Metadata is later used for:

- retrieval tracking
- reranking
- context construction
- citations
- debugging

The `chunk_id` is stable and follows:

```text
document_id:chunk_number
```

Example:

```text
company-policy:chunk_000008
```

---

# 12. Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings convert text into numerical vector representations.

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

Example:

```text
"What are the confidentiality duties of directors?"
                     ↓
             Embedding Vector
```

Semantically similar text should have similar vector representations.

The embedding model is used during indexing and semantic retrieval.

Relevant module:

```text
app/embeddings/embedder.py
```

---

# 13. FAISS Dense Retrieval

FAISS is used for dense vector retrieval.

During indexing:

```text
Chunk
 ↓
Embedding
 ↓
FAISS
```

During querying:

```text
Question
 ↓
Question Embedding
 ↓
FAISS Similarity Search
 ↓
Candidate Chunks
```

Dense retrieval is useful when the query and document use different wording but have similar meanings.

Example:

```text
Query:
"How should directors protect confidential information?"

Document:
"Directors shall safeguard the confidentiality
of all information received by virtue of their position."
```

The wording differs, but the semantic meaning is related.

Relevant module:

```text
app/vectorstore/faiss_store.py
```

---

# 14. BM25 Sparse Retrieval

BM25 is a keyword-based information retrieval algorithm.

It is useful when exact terms matter.

Example:

```text
Query:
"unpublished price sensitive information"
```

If those terms appear in a document, BM25 can identify the relevant document effectively.

BM25 complements dense retrieval.

Relevant module:

```text
app/retrieval/bm25_retriever.py
```

---

# 15. Hybrid Retrieval

The project combines:

```text
FAISS
+
BM25
```

The two approaches have different strengths.

| Retrieval Method | Strength |
|---|---|
| FAISS | Semantic similarity |
| BM25 | Exact keyword matching |

The query is sent to both systems:

```text
                  Query
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        FAISS                BM25
     Semantic              Keyword
     Retrieval             Retrieval
          │                   │
          └─────────┬─────────┘
                    ▼
                RRF Fusion
```

This produces a stronger candidate set than relying on only one retrieval method.

---

# 16. Reciprocal Rank Fusion

FAISS and BM25 use different scoring mechanisms.

Their raw scores should not simply be added together.

The project uses Reciprocal Rank Fusion (RRF).

Basic formula:

```text
RRF Score = Σ 1 / (k + rank)
```

The important idea is:

> A document appearing near the top of multiple retrieval systems should receive a stronger combined ranking.

Example:

```text
FAISS:

1. Chunk A
2. Chunk B
3. Chunk C


BM25:

1. Chunk B
2. Chunk A
3. Chunk D
```

After RRF:

```text
Chunk A → Strong
Chunk B → Strong
Chunk C → Weaker
Chunk D → Weaker
```

RRF combines rankings rather than assuming raw scores from different retrieval systems are directly comparable.

---

# 17. Cross-Encoder Reranking

After hybrid retrieval, candidate documents are reranked using a Cross-Encoder.

Current model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder receives:

```text
Query
+
Document
```

and predicts relevance.

Conceptually:

```text
Query
 +
Candidate Document
        ↓
Cross-Encoder
        ↓
Relevance Score
```

The reranker input also includes structural information such as:

```text
Section
Page
Content
```

The architecture is:

```text
Initial Retrieval
        ↓
Candidate Generation
        ↓
Reranking
        ↓
Best Candidates
```

Relevant module:

```text
app/retrieval/reranker.py
```

---

# 18. Context Expansion

Sometimes a retrieved chunk does not contain enough surrounding information.

The project supports neighboring chunk expansion.

Example:

```text
Chunk 7
Chunk 8 ← Directly retrieved
Chunk 9
```

With:

```text
expand_neighbors = 1
```

the system can provide:

```text
Chunk 7 → Supporting context
Chunk 8 → Direct evidence
Chunk 9 → Supporting context
```

Direct and expanded chunks are explicitly distinguished through metadata:

```text
retrieval_type = "direct"
```

or:

```text
retrieval_type = "expanded"
```

Relevant module:

```text
app/retrieval/context_expander.py
```

---

# 19. Context Builder

The Context Builder prepares selected chunks for the LLM.

It handles:

- metadata formatting
- chunk formatting
- token usage
- context limits
- selected documents

Example:

```text
[Document: company-policy]
[Section: III. Duties of Independent Directors]
[Pages: 4]
[Chunk: company-policy:chunk_000008]
[Retrieval Type: direct]
[Reranker Score: 8.5699]

[Content]

not disclose confidential information...
```

The current MVP uses an approximate token calculation:

```text
4 characters ≈ 1 token
```

This is sufficient for the MVP but should eventually be replaced with model-specific tokenization.

Relevant modules:

```text
app/generation/context_builder.py
app/generation/token_counter.py
```

---

# 20. Conversation History

The application supports multi-turn conversations.

Conversation history is separate from the document knowledge base.

Example:

```text
User:
What are the duties of independent directors?

Assistant:
...

User:
What about confidential information?
```

The second question can use the previous conversation to understand references such as "they" or "their".

The current MVP stores conversation history locally as JSON.

Example:

```text
data/conversations/
```

Conversation messages are stored with roles such as:

```json
{
    "role": "user",
    "content": "What are the duties of independent directors?"
}
```

and:

```json
{
    "role": "assistant",
    "content": "..."
}
```

Important concept:

```text
Conversation History
        ≠
Knowledge Base
```

Conversation history provides conversational context.

Retrieved documents provide factual evidence.

Relevant module:

```text
app/conversation/history.py
```

---

# 21. Prompt Builder

The Prompt Builder combines the information required by the LLM.

Logical components:

```text
System Instructions
        +
Conversation History
        +
Retrieved Context
        +
Current User Question
```

The system prompt instructs the model to:

1. Use retrieved context as the primary source of truth.
2. Avoid inventing unsupported information.
3. Say when information is unavailable.
4. Provide concise answers.
5. Use document metadata when appropriate.
6. Use conversation history to understand follow-up questions.
7. Not treat conversation history as authoritative evidence when it conflicts with retrieved context.

Conceptually:

```text
System Message
      ↓
Conversation History
      ↓
Retrieved Context
      ↓
Current Question
      ↓
LLM
```

Relevant module:

```text
app/generation/prompt_builder.py
```

---

# 22. Local LLM Generation

The application uses Ollama for local LLM inference.

Current model:

```text
gemma3:4b
```

The model is accessed through LangChain's Ollama integration.

Flow:

```text
Prompt
 ↓
Ollama
 ↓
Gemma 3 4B
 ↓
Generated Answer
```

Current temperature:

```text
0.0
```

A low temperature is useful for a factual RAG application because the goal is to produce consistent answers rather than highly creative responses.

Relevant module:

```text
app/generation/llm.py
```

---

# 23. Citation Builder

Citations are generated separately from the LLM.

This is an important architectural decision.

The LLM should not be responsible for inventing:

```text
document name
page number
chunk ID
section
```

Instead, the application uses metadata from retrieved documents.

Example:

```text
[1] — company-policy — III. Duties of Independent Directors:
    — Page(s) 4
    — Chunk company-policy:chunk_000008
```

Expanded chunks are marked as:

```text
Supporting context
```

The citation flow is:

```text
Retrieved Document
        ↓
Document Metadata
        ↓
Citation Builder
        ↓
Citation
```

not:

```text
Retrieved Document
        ↓
LLM
        ↓
Invented Citation
```

Relevant module:

```text
app/generation/citation_builder.py
```

---

# 24. RAG Pipeline

`RAGPipeline` is the central orchestrator.

Instead of the CLI directly calling every component, it calls:

```python
response = rag.ask(question)
```

The pipeline internally coordinates:

```text
Conversation History
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Context Expansion
        ↓
Context Builder
        ↓
Prompt Builder
        ↓
LLM
        ↓
Citation Builder
        ↓
Conversation Storage
```

The response has the structure:

```python
{
    "answer": "...",
    "citations": [...],
    "documents": [...]
}
```

Relevant module:

```text
app/pipeline/rag_pipeline.py
```

This layer transforms individual RAG components into one application-level workflow.

---

# 25. Conversational CLI

The project provides an interactive command-line interface.

Start the application:

```powershell
python -m app.cli.app
```

Example:

```text
========================================
       Enterprise RAG Assistant
========================================

Document : company-policy.pdf
Model    : gemma3:4b

Commands:
  exit  - Exit the application
  clear - Clear conversation

RAG system ready.

You: What are the duties of independent directors?

Assistant:
Independent directors must...

Sources:
[1] — company-policy — Page(s) 4
[2] — company-policy — Page(s) 4 — Supporting context

You:
```

Supported commands:

```text
exit
```

or:

```text
quit
```

to exit.

Use:

```text
clear
```

to clear the current conversation history.

---

# 26. Configuration

Configuration is centralized in:

```text
app/config.py
```

Example:

```python
RAGConfig(
    pdf_path="data/company-policy.pdf",
    vector_store_path="data/faiss_index",
    llm_model="gemma3:4b",
    temperature=0.0,
    max_context_tokens=4000,
    retrieval_k=10,
    candidate_k=10,
    final_k=1,
    expand_neighbors=1,
    conversation_id="cli_conversation",
    conversation_storage_dir="data/conversations",
)
```

Important configuration values:

| Configuration | Purpose |
|---|---|
| `pdf_path` | PDF used as the knowledge source |
| `vector_store_path` | FAISS storage location |
| `llm_model` | Ollama model |
| `temperature` | LLM generation temperature |
| `max_context_tokens` | Maximum context size |
| `retrieval_k` | Number of initial retrieval results |
| `candidate_k` | Number of RRF candidates |
| `final_k` | Number of reranked direct results |
| `expand_neighbors` | Number of neighboring chunks |
| `conversation_id` | Conversation identifier |
| `conversation_storage_dir` | Conversation storage location |

The CLI reads these values dynamically rather than hardcoding the document and model into the interface.

---

# 27. End-to-End Example

Consider:

```text
What are the duties of independent directors regarding confidential information?
```

The complete process is:

```text
Question
 ↓
FAISS
 ↓
BM25
 ↓
RRF
 ↓
Cross-Encoder
 ↓
Context Expansion
 ↓
Context Builder
 ↓
Prompt Builder
 ↓
Gemma
```

The answer may be:

```text
Independent directors must not disclose confidential information,
including commercial secrets, technologies, advertising and sales
promotion plans, and unpublished price-sensitive information unless
disclosure is expressly approved by the Board or required by law.
```

The Citation Builder then produces:

```text
Sources:

[1] — company-policy — III. Duties of Independent Directors:
    — Page(s) 4
    — Chunk company-policy:chunk_000008

[2] — company-policy — III. Duties of Independent Directors:
    — Page(s) 4
    — Chunk company-policy:chunk_000007
    — Supporting context
```

---

# 28. Testing

The project contains tests for individual components and integration.

The purpose of these tests is to verify each part independently before relying on the complete application.

## Document ingestion

```powershell
python -m tests.test_docling
```

## BM25

```powershell
python -m tests.test_bm25
```

## Hybrid retrieval

```powershell
python -m tests.test_hybrid
```

## Reranking

```powershell
python -m tests.test_reranker
```

## Context expansion

```powershell
python -m tests.test_context_expansion
```

## Context builder

```powershell
python -m tests.test_context_builder
```

## Prompt builder

```powershell
python -m tests.test_prompt_builder
```

## Conversation history

```powershell
python -m tests.test_conversation_history
```

## Local LLM

```powershell
python -m tests.test_llm
```

## Citation builder

```powershell
python -m tests.test_citation_builder
```

## End-to-end generation

```powershell
python -m tests.test_rag_generation
```

## RAG Pipeline

```powershell
python -m tests.test_pipeline
```

The final pipeline test verifies that the individual modules work together through `RAGPipeline`.

---

# 29. Current MVP

The current MVP implements:

- [x] PDF ingestion
- [x] Docling parsing
- [x] Structural/hybrid chunking
- [x] Stable document IDs
- [x] Stable chunk IDs
- [x] Page metadata
- [x] Heading metadata
- [x] Sentence Transformer embeddings
- [x] FAISS dense retrieval
- [x] BM25 sparse retrieval
- [x] Hybrid retrieval
- [x] Reciprocal Rank Fusion
- [x] Cross-Encoder reranking
- [x] Context expansion
- [x] Context budgeting
- [x] Conversation history
- [x] Prompt construction
- [x] Local Ollama inference
- [x] Gemma 3 4B
- [x] Deterministic citations
- [x] RAG pipeline orchestration
- [x] Interactive CLI
- [x] Basic unanswerable-question handling

---

# 30. Current Limitations

This is an MVP and is not production-ready.

## 30.1 Indexing happens during application startup

Currently, starting the application can perform:

```text
PDF
 ↓
Docling
 ↓
Chunking
 ↓
Embedding
 ↓
FAISS
```

A production architecture should separate:

```text
Offline Indexing
```

from:

```text
Online Querying
```

---

## 30.2 Single-document workflow

The current MVP is primarily designed around a single PDF.

A future version should support multiple documents:

```text
Documents
 ├── company-policy.pdf
 ├── employee-handbook.pdf
 ├── security-policy.pdf
 └── leave-policy.pdf
```

with document-level filtering and metadata.

---

## 30.3 Local JSON conversation storage

The MVP stores conversation history as local JSON.

A production application may use:

```text
PostgreSQL
Redis
Document Database
```

depending on requirements.

---

## 30.4 Basic grounding protection

The prompt instructs the LLM not to invent information.

The MVP has been tested using an unanswerable question and the model responded that the information was not available in the provided documents.

However, a production system should include a dedicated grounding and answerability evaluation mechanism.

---

## 30.5 No authentication

The MVP does not currently implement:

```text
Authentication
Authorization
User Management
Document Access Control
```

---

## 30.6 No production observability

The MVP does not yet provide comprehensive:

```text
Logging
Metrics
Tracing
Latency Monitoring
Token Usage Monitoring
Retrieval Quality Monitoring
Answer Quality Monitoring
```

---

## 30.7 CLI only

The current application uses a command-line interface.

A future version can expose the RAG pipeline through an API and web interface.

---

# 31. Future Improvements

Future versions can evolve the MVP into a production-grade RAG system.

## Indexing

```text
Persistent index lifecycle
Multi-document indexing
Incremental indexing
Document updates
Document deletion
Index versioning
```

## Retrieval

```text
Better retrieval evaluation
Metadata filtering
Query rewriting
Query expansion
Hybrid retrieval tuning
Adaptive top-k
```

## Generation

```text
Answerability detection
Grounding verification
Claim verification
Citation validation
Better prompt strategies
```

## Storage

```text
Persistent database
Persistent conversation storage
Document metadata database
```

## API

```text
FastAPI
REST API
Streaming responses
Authentication
Authorization
```

## Observability

```text
Structured logging
Metrics
Tracing
Latency tracking
Token usage tracking
Retrieval monitoring
```

## Evaluation

```text
Recall@K
Precision@K
MRR
nDCG
Context relevance
Context precision
Context recall
Answer relevance
Faithfulness
Groundedness
```

## Deployment

```text
Cloud infrastructure
Containerization
Docker
GPU inference
Scalable model serving
Production monitoring
```

---

# 32. Important RAG Concepts

This section is intended as a revision guide.

## 32.1 Embedding ≠ Retrieval

Embedding:

```text
Text
 ↓
Vector
```

Retrieval:

```text
Query
 ↓
Relevant Documents
```

Embeddings are one mechanism used by dense retrieval.

---

## 32.2 Vector Similarity ≠ Relevance

A vector database finds mathematically similar vectors.

That does not guarantee that the retrieved document completely answers the question.

Therefore:

```text
Embedding Retrieval
        ↓
Candidate Retrieval
        ↓
Reranking
```

can improve relevance.

---

## 32.3 Dense and Sparse Retrieval

Dense retrieval:

```text
Meaning / Semantics
```

Sparse retrieval:

```text
Words / Keywords
```

FAISS is used for dense retrieval.

BM25 is used for sparse retrieval.

---

## 32.4 Why Hybrid Retrieval?

Dense retrieval may miss exact terminology.

Sparse retrieval may miss semantic similarity.

Combining them provides complementary retrieval signals.

```text
FAISS
 +
BM25
 ↓
Hybrid Retrieval
```

---

## 32.5 Why RRF?

FAISS and BM25 have different scoring systems.

RRF combines their rankings without requiring their raw scores to be directly comparable.

---

## 32.6 Retrieval vs Reranking

Retrieval is usually optimized for:

```text
Recall
```

Reranking is used to improve:

```text
Precision
```

General architecture:

```text
Retrieve many candidates
        ↓
Rerank candidates
        ↓
Keep the best candidates
```

---

## 32.7 Direct vs Expanded Context

Direct:

```text
Retrieved because it matched the query.
```

Expanded:

```text
Added because it is near a relevant chunk.
```

Therefore:

```text
Direct ≠ Expanded
```

Direct chunks are primary evidence.

Expanded chunks are supporting context.

---

## 32.8 Conversation History ≠ Knowledge Base

Conversation history helps understand:

```text
"What about their responsibilities?"
```

The knowledge base provides factual evidence.

Therefore:

```text
Conversation History
        ≠
Knowledge Base
```

---

## 32.9 Retrieved Context ≠ Evidence

A retriever can return documents even when the answer is not present.

Therefore:

```text
Retrieved Document
        ≠
Answer Exists
```

This is one of the most important concepts in RAG.

---

## 32.10 Citation ≠ Grounding

A response can contain a citation without actually being supported by that source.

Example:

```text
Answer:
The company provides 26 weeks of maternity leave.

Citation:
company-policy, Page 4
```

The citation exists, but the document might not contain anything about maternity leave.

Therefore:

```text
Citation
   ≠
Grounding
```

A production RAG system needs to verify that the answer is actually supported by the cited evidence.

---

## 32.11 Retrieval Score ≠ Confidence

A reranker score such as:

```text
8.56
```

should not automatically be interpreted as:

```text
85.6% confidence
```

Retrieval and reranking scores are primarily useful for ranking candidates.

---

## 32.12 Context Budget Matters

The LLM has a finite context window.

Therefore:

```text
More retrieved documents
        ≠
Better answer
```

Too much irrelevant context can reduce answer quality.

The goal is:

```text
Small
+
Relevant
+
Sufficient
Context
```

---

# 33. RAG Revision Cheat Sheet

When revising the project, remember this indexing pipeline:

```text
                    DOCUMENT
                       │
                       ▼
                   Docling
                       │
                       ▼
                    Chunking
                       │
                       ▼
                   Metadata
                       │
                       ▼
                   Embeddings
                       │
                       ▼
                     FAISS
```

Then at query time:

```text
                     QUERY
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           FAISS                BM25
             │                   │
             └─────────┬─────────┘
                       ▼
                      RRF
                       │
                       ▼
                  Reranking
                       │
                       ▼
                Context Expansion
                       │
                       ▼
                 Context Builder
                       │
                       ▼
                Prompt Builder
                       │
                       ▼
                     LLM
                       │
                       ▼
                    Answer
                       │
                       ▼
                 Citations
```

Remember:

```text
FAISS
    → semantic retrieval

BM25
    → keyword retrieval

RRF
    → combines rankings

Cross-Encoder
    → reranks candidates

Context Expansion
    → adds neighboring context

Context Builder
    → controls what enters the LLM

Prompt Builder
    → constructs the LLM input

LLM
    → generates the answer

Citation Builder
    → identifies source metadata

Conversation History
    → maintains multi-turn context
```

---

# 34. Learning Roadmap

## Level 1 — Document Processing

Learn:

```text
PDF parsing
Document structure
Chunking
Metadata
```

## Level 2 — Embeddings

Learn:

```text
Embeddings
Vector representations
Cosine similarity
Semantic similarity
```

## Level 3 — Retrieval

Learn:

```text
Dense retrieval
Sparse retrieval
BM25
FAISS
Hybrid retrieval
```

## Level 4 — Retrieval Quality

Learn:

```text
RRF
Reranking
Cross-Encoders
Top-K
Recall
Precision
MRR
nDCG
```

## Level 5 — Context and Generation

Learn:

```text
Context windows
Token budgets
Prompt construction
Conversation history
Temperature
Local LLM inference
```

## Level 6 — RAG Quality

Learn:

```text
Answerability
Groundedness
Faithfulness
Context relevance
Answer relevance
Citation correctness
Hallucination detection
```

## Level 7 — Production RAG

Learn:

```text
Persistent indexing
Multiple documents
Metadata filtering
Authentication
Authorization
Caching
Observability
Evaluation
Scaling
Deployment
```

---

# 35. Version

Current version:

```text
Version: 0.1.0
Status: MVP
```

The current version represents the first complete working conversational RAG application.

Future versions can focus on production hardening, evaluation, scalability, and deployment.

---

# Core Principle

The most important concept behind this project is:

```text
              RAG
               │
       ┌───────┴───────┐
       │               │
   RETRIEVAL       GENERATION
       │               │
       ▼               ▼
 Find Evidence     Use Evidence
       │               │
       ▼               ▼
 FAISS + BM25       Prompt
       │               │
       ▼               ▼
      RRF              LLM
       │               │
       ▼               ▼
  Reranking          Answer
       │               │
       ▼               ▼
Context Expansion   Citations
       │
       ▼
Context Builder
```

The core philosophy is:

> **Retrieval finds the evidence. Generation uses the evidence. Citations identify the evidence.**

The LLM should not be treated as the source of truth.

The retrieved documents should be treated as the factual source for the RAG response.
