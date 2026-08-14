# Advanced Agentic RAG & Multi-Agent Research System

A local, zero-cost research system that combines **Advanced RAG, LangGraph, and specialized multi-agent orchestration** to retrieve, analyze, write, and review evidence-grounded research responses.

This project is designed as **Project 7** in a structured GenAI interview-preparation roadmap.

---

# 1. Project Objective

The primary objective is to learn and demonstrate:

- Advanced RAG
- Hybrid retrieval
- Query rewriting
- Metadata filtering
- Reranking
- Agents
- Planning
- Information gathering
- Multi-agent orchestration
- LangGraph state management
- Conditional workflow routing
- Evidence-grounded research

The project intentionally avoids rebuilding concepts already covered in Projects 1–6.

The focus is:

> **Advanced RAG + Agents + LangGraph + Multi-Agent Orchestration**

The implementation remains local, manageable, testable, and interview-focused.

---

# 2. Problem Statement

Basic RAG can retrieve useful information, but retrieval quality can degrade when:

- the user's wording differs from the source wording;
- exact technical terms are important;
- semantic retrieval alone produces imperfect candidates;
- keyword retrieval alone misses semantic relationships;
- retrieved candidates need stronger relevance scoring;
- research requires multiple specialized responsibilities;
- an application needs explicit workflow state and controlled routing.

This project addresses these problems by combining:

```text
Query Rewriting
      ↓
Hybrid Retrieval
      ↓
Metadata Filtering
      ↓
Reranking
      ↓
Evidence
      ↓
Multi-Agent Research Workflow
```

---

# 3. Project Scope

## Core Concepts

The project implements:

### Advanced RAG

- Vector retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Metadata-aware retrieval
- Query rewriting
- Reranking

### Agents

- Researcher
- Analyst
- Writer
- Reviewer

### LangGraph

- State
- Nodes
- Edges
- Conditional routing
- Workflow orchestration

### Quality

- Input validation
- Evidence validation
- Reviewer validation
- Failure handling
- Unit tests
- Integration tests
- End-to-end testing

---

# 4. Target Architecture

```text
                    User Question
                          |
                          v
                  LangGraph Workflow
                          |
                          v
                  Planner / Preparation
                          |
                          v
                 Query Rewriting
                          |
                          v
                  Researcher Agent
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Vector Retrieval          BM25 Retrieval
              |                       |
              +-----------+-----------+
                          |
                          v
                  Hybrid / RRF Fusion
                          |
                          v
                  Metadata Filtering
                          |
                          v
                 CrossEncoder Reranking
                          |
                          v
                       Evidence
                          |
                          v
                   Analyst Agent
                          |
                          v
                    Writer Agent
                          |
                          v
                   Reviewer Agent
                          |
                  +-------+-------+
                  |               |
               Approved         Revision
                  |               |
                  v               |
                 END <-------------+
```

---

# 5. End-to-End Flow

A research request moves through the system as follows:

```text
1. User submits a research question
                  ↓
2. Planner validates and prepares the task
                  ↓
3. Query rewriting improves retrieval formulation
                  ↓
4. Researcher gathers candidate evidence
                  ↓
5. Vector search finds semantic matches
                  ↓
6. BM25 finds lexical matches
                  ↓
7. RRF combines both result sets
                  ↓
8. Metadata filtering restricts candidates when required
                  ↓
9. CrossEncoder reranks candidates
                  ↓
10. Analyst synthesizes evidence
                  ↓
11. Writer creates the research report
                  ↓
12. Reviewer validates the report
                  ↓
13. LangGraph routes approved/rejected output
```

---

# 6. Advanced RAG Pipeline

## 6.1 Document Loading

The system loads a local JSON corpus containing structured source documents.

Each source document contains information such as:

- document ID
- title
- source
- category
- text
- metadata

---

## 6.2 Chunking

Source documents are divided into deterministic overlapping chunks.

The current chunker uses:

```text
chunk_size = 80 words
overlap    = 15 words
```

The goal is to preserve enough context between adjacent chunks while keeping retrieval units manageable.

---

# 7. Vector Retrieval

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model creates dense vector representations of document chunks.

The same embedding model is used to encode the query.

Semantic similarity is then used to identify relevant chunks.

### Why vector retrieval?

Vector retrieval can find semantically related information even when the query and source use different wording.

Example:

```text
Query:
"How can systems find information when wording differs?"

Source:
"Semantic representations allow retrieval despite lexical variation."
```

The wording is different, but the underlying meaning is related.

---

# 8. BM25 Keyword Retrieval

BM25 provides lexical retrieval.

It is particularly useful when the query contains:

- technical terminology
- exact phrases
- identifiers
- domain-specific words

For example:

```text
Query:
"What does BM25 do?"
```

BM25 can directly leverage the occurrence of the term:

```text
BM25
```

---

# 9. Hybrid Retrieval

Vector and BM25 retrieval have complementary strengths.

```text
Vector Search
    ↓
Semantic similarity

BM25
    ↓
Exact lexical matching
```

Instead of relying on only one retrieval method, the project combines their ranked results using:

> **Reciprocal Rank Fusion (RRF)**

Conceptually:

```text
Vector ranking
      +
BM25 ranking
      ↓
RRF
      ↓
Combined candidate ranking
```

### Why hybrid retrieval?

Vector search is strong when wording differs.

BM25 is strong when exact terminology matters.

Combining them improves retrieval robustness because the two approaches fail differently.

---

# 10. Metadata Filtering

Document metadata is preserved throughout the retrieval pipeline.

Example:

```text
category = reranking
```

A query can therefore be restricted to documents belonging to a particular category.

Metadata filtering is useful when:

- the corpus contains multiple domains;
- the query specifies a topic;
- retrieval should be constrained;
- irrelevant categories should be removed.

---

# 11. Reranking

Initial retrieval is designed for **broad candidate discovery**.

The project then uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

for candidate reranking.

The CrossEncoder receives the query and candidate document together and produces a relevance score.

```text
Query
  +
Candidate Document
      ↓
CrossEncoder
      ↓
Relevance Score
```

### Why rerank?

Initial retrieval methods are optimized for efficient candidate discovery.

CrossEncoder scoring is more expensive.

Therefore:

```text
Large corpus
    ↓
Fast retrieval
    ↓
Small candidate set
    ↓
Expensive reranking
    ↓
High-quality ordering
```

This is more practical than applying expensive relevance scoring to every document.

---

# 12. Query Rewriting

The system can use:

```text
Qwen2.5-0.5B-Instruct
```

locally for retrieval-query rewriting.

The purpose is to transform a natural-language research question into a retrieval-oriented formulation.

Example:

```text
Original:
How can a RAG system retrieve useful information
when the user's wording differs from the source documents?

Rewritten:
RAG retrieval with different query and document wording
```

---

# 13. Query Rewrite Failure Handling

Query rewriting is a generative operation and therefore cannot automatically be trusted.

A small local model can:

- over-expand a query;
- change the original intent;
- introduce unrelated concepts;
- produce a poor retrieval formulation.

Therefore the system treats rewriting as an optional optimization rather than an unquestionable transformation.

The original query remains the safe fallback.

Conceptually:

```text
Original Query
      |
      v
Query Rewriter
      |
      v
Validate Rewrite
   /         \
valid       invalid
 |             |
 v             v
rewrite      original
```

This prevents a poor generated rewrite from destroying retrieval quality.

---

# 14. Multi-Agent Architecture

The system separates research responsibilities into four specialized agents.

```text
Researcher
    ↓
Analyst
    ↓
Writer
    ↓
Reviewer
```

Each agent has a clear responsibility.

---

# 15. Researcher Agent

The Researcher is responsible for:

- receiving the research task;
- executing the retrieval process;
- gathering evidence;
- returning ranked evidence to the workflow.

The Researcher does not write the final report.

This separation keeps retrieval and generation responsibilities distinct.

---

# 16. Analyst Agent

The Analyst receives retrieved evidence.

Responsibilities:

- inspect retrieved evidence;
- organize evidence;
- synthesize retrieved information;
- prepare material for report generation.

The Analyst currently uses deterministic Python logic rather than an additional LLM generation.

This is intentional.

Not every agent operation requires an LLM call.

---

# 17. Writer Agent

The Writer converts the analysis into a structured research report.

The current report structure contains:

```text
# Research Report

## Question

## Analysis
```

The Writer is responsible for presentation rather than retrieval.

---

# 18. Reviewer Agent

The Reviewer validates the generated report.

The current reviewer checks:

- draft exists;
- evidence exists;
- required report sections exist;
- recognizable retrieved evidence appears in the report.

The reviewer returns:

```text
review message
+
approved / rejected
```

This keeps quality validation explicit.

---

# 19. Why Deterministic Agents?

It may seem that every agent should call an LLM.

That is not necessary.

The project deliberately uses deterministic Python for operations that do not require generation:

- validation
- routing
- formatting
- evidence handling
- structural checks

This provides:

- lower latency;
- predictable behavior;
- easier testing;
- easier debugging;
- lower CPU usage.

The local Qwen model is used where it demonstrates a distinct GenAI concept: query rewriting.

---

# 20. LangGraph

LangGraph is used to orchestrate the research workflow.

The graph contains:

- shared state;
- nodes;
- edges;
- conditional routing.

---

# 21. Research State

The workflow state carries information between nodes.

Conceptually:

```text
ResearchState
├── question
├── retrieval_query
├── evidence
├── analysis
├── draft
├── review
├── approved
└── revision_count
```

This makes the workflow explicit and testable.

---

# 22. LangGraph Nodes

The main workflow responsibilities are represented as nodes:

```text
Planner
Researcher
Analyst
Writer
Reviewer
```

Each node reads relevant state and returns state updates.

---

# 23. Conditional Routing

The Reviewer determines whether the report passes validation.

Conceptually:

```text
Reviewer
   |
   +---- approved ----> END
   |
   +---- rejected ----> revision
```

Conditional routing demonstrates how LangGraph can control multi-step workflows rather than simply executing a fixed sequence.

---

# 24. Why LangGraph?

A normal Python function chain could execute the same steps.

LangGraph provides additional value when workflows require:

- explicit state;
- conditional routing;
- iterative execution;
- branching;
- agent coordination;
- inspectable workflow structure.

For this project, LangGraph is primarily being used to learn and demonstrate stateful agent orchestration.

---

# 25. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Environment | UV |
| LLM | Qwen2.5-0.5B-Instruct |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Hybrid Fusion | Reciprocal Rank Fusion |
| Reranking | CrossEncoder |
| Orchestration | LangGraph |
| Data Validation | Pydantic |
| Testing | pytest |
| Model Runtime | PyTorch CPU |
| Model Hub | Hugging Face |
| Version Control | Git |
| Repository | GitHub |
| Cost | ₹0 |

---

# 26. Models Used

## Qwen2.5-0.5B-Instruct

Purpose:

```text
Query rewriting
```

Local model execution is used to maintain the project at zero inference cost.

---

## all-MiniLM-L6-v2

Purpose:

```text
Dense embeddings
```

Observed embedding dimension:

```text
384
```

---

## cross-encoder/ms-marco-MiniLM-L6-v2

Purpose:

```text
Candidate reranking
```

It is applied after initial retrieval.

---

# 27. Local Model Strategy

Existing Hugging Face cache infrastructure is reused.

The project uses local cached models rather than downloading models repeatedly.

Relevant cached models include:

```text
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
all-MiniLM-L6-v2
cross-encoder/ms-marco-MiniLM-L6-v2
```

The 0.5B Qwen model is preferred for this project because CPU inference is significantly more practical than using a larger local model.

---

# 28. Repository Structure

```text
advanced-agentic-rag-multi-agent-research-system/
│
├── data/
│   ├── documents/
│   │   └── corpus.json
│   └── index/
│
├── docs/
│   └── challenges_failure_analysis.md
│
├── scratch/
│
├── src/
│   └── agentic_rag/
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── analyst.py
│       │   ├── researcher.py
│       │   ├── reviewer.py
│       │   └── writer.py
│       │
│       ├── config/
│       │
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── nodes.py
│       │   ├── researcher.py
│       │   ├── state.py
│       │   └── workflow.py
│       │
│       ├── llm/
│       │
│       ├── models/
│       │   ├── document.py
│       │   └── source_document.py
│       │
│       ├── rag/
│       │   ├── chunker.py
│       │   ├── loader.py
│       │   └── retrieval components
│       │
│       └── utils/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

---

# 29. Installation

## Requirements

- Windows
- Python 3.12
- UV
- Git
- Local Hugging Face model cache

Verify Python:

```powershell
uv run python --version
```

Expected:

```text
Python 3.12.x
```

Install/synchronize dependencies:

```powershell
uv sync
```

---

# 30. Running Tests

Run the complete suite:

```powershell
uv run pytest -q
```

The final validated suite contains:

```text
25 passed
```

The project separates deterministic unit tests from real-model integration tests.

This keeps normal tests fast while still proving that the actual local models work.

---

# 31. Test Strategy

Testing is divided into multiple layers.

## Unit Tests

Test:

- data models;
- chunking;
- retrieval logic;
- agents;
- reviewer behavior;
- failure handling;
- graph components.

## Integration Tests

Real local models are used for selected integration tests.

Examples:

- real MiniLM embeddings;
- real CrossEncoder reranking;
- real Qwen query rewriting;
- real retrieval;
- complete end-to-end workflow.

## End-to-End Test

The complete workflow is validated using a real research question.

Example:

```text
How does hybrid retrieval improve the quality of RAG systems?
```

The test verifies:

```text
retrieval query
evidence
analysis
draft
review
approval
```

---

# 32. Validation Results

Final automated validation:

```text
25 passed
```

The end-to-end workflow was also manually validated.

Example result:

```text
Evidence Count: 3

Review:
Approved: report contains retrieved evidence
and required report sections.

Approved:
True
```

---

# 33. Performance

The project runs locally on CPU.

This is intentionally a zero-cost architecture.

## Qwen Performance Baseline

Observed warm generation:

```text
Generated tokens: 64
Time: approximately 8.88 seconds
Tokens/sec: approximately 7.21
```

Observed first generation:

```text
Generated tokens: 64
Time: approximately 15.15 seconds
Tokens/sec: approximately 4.23
```

Query rewriting was observed at approximately:

```text
8 seconds warm rewrite latency
```

The exact latency depends on:

- CPU load;
- prompt size;
- generated token count;
- model loading;
- operating-system state.

---

# 34. Performance Design Decisions

Because CPU inference is relatively slow:

- Qwen is not called for every agent;
- deterministic logic is preferred where possible;
- real-model tests are isolated;
- model caches are reused;
- generation length is constrained;
- expensive reranking happens only after candidate retrieval.

---

# 35. Cost

The project is:

```text
₹0
```

No paid APIs are required.

No paid model inference is required.

No paid GPU is required.

No paid hosting is required.

The project uses:

- local models;
- open-source libraries;
- local retrieval;
- local CPU inference;
- Git/GitHub.

---

# 36. Challenges / Problems / Drawbacks / Failure Analysis

A detailed failure analysis is maintained in:

```text
docs/challenges_failure_analysis.md
```

The analysis follows the required pattern:

```text
Symptom
   ↓
Root Cause
   ↓
Fix / Workaround
   ↓
Lesson Learned
   ↓
Prevention
```

Major Project 7 issues included:

1. PowerShell multiline here-string problems
2. Initial src-layout import failure
3. Duplicate pytest module names
4. Query rewriting semantic drift
5. CPU inference latency
6. Hugging Face Windows symlink/cache warning
7. CrossEncoder model download
8. Circular imports between agents and graph
9. Reviewer test-contract mismatch
10. Temporary exploratory scripts
11. Git checkpoint management
12. Complex PowerShell/Python command quoting
13. Retrieval versus reranking relevance differences
14. Avoiding unnecessary LLM calls

---

# 37. Important Engineering Lessons

## Reuse Before Reinstall

Existing model caches and proven infrastructure should be reused whenever possible.

## Measure Before Optimizing

Actual CPU latency is more useful than assumptions about model performance.

## Deterministic First

Do not turn every operation into an LLM call.

## Expensive Work Should Be Isolated

Real model inference belongs in a small number of integration tests.

## Warnings Are Not Automatically Failures

A warning should be investigated for actual impact before changing the architecture.

## Tests Are Executable Contracts

When architecture changes, tests should be deliberately updated to represent the new intended contract.

## Keep Git Checkpoints

Use:

```text
edit
→ test
→ git status
→ git add specific files
→ git status
→ commit
→ push
→ git status
```

---

# 38. Circular Import Lesson

One important Project 7 failure involved:

```text
agents
   ↓
graph
   ↓
workflow
   ↓
nodes
   ↓
agents
```

This created a circular dependency.

The solution was to keep package-level imports lightweight and lazily import the workflow when it is actually requested.

Lesson:

> Package `__init__.py` files should avoid unnecessary eager imports, especially across mutually dependent architecture layers.

---

# 39. Query Rewriting Limitation

Query rewriting is not guaranteed to improve retrieval.

A small local model can introduce semantic drift.

For example, the model may interpret a wording-related retrieval question as a language-related question.

Therefore:

```text
Generated Rewrite
      ↓
Validation
      ↓
Accept
   OR
Fallback to Original
```

The original query is always available as the safe fallback.

---

# 40. Reranking Limitation

Reranking improves candidate ordering but does not guarantee factual correctness.

A high relevance score means:

```text
The candidate appears relevant to the query.
```

It does not mean:

```text
The candidate is factually correct.
```

Reranking should therefore be considered a retrieval-quality component rather than a truth detector.

---

# 41. Reviewer Limitation

The current Reviewer performs deterministic structural/evidence checks.

It does not provide full semantic evaluation.

Approval does not guarantee:

- factual correctness;
- complete reasoning;
- absence of hallucinations;
- perfect evidence attribution.

This limitation is intentional for Project 7.

---

# 42. Why Not Rebuild Basic RAG?

Basic RAG was already implemented in Project 2.

Project 7 therefore focuses on the advanced retrieval problems that appear after basic RAG:

```text
Basic RAG
   ↓
Advanced Retrieval
   ↓
Hybrid Search
   ↓
Filtering
   ↓
Reranking
   ↓
Query Rewriting
   ↓
Agentic Research
   ↓
Multi-Agent Orchestration
```

This avoids duplicating previous learning.

---

# 43. Why Not Use External Web Search?

External web search was intentionally kept outside the core Project 7 implementation.

The project focuses on learning:

- Advanced RAG;
- agents;
- LangGraph;
- multi-agent orchestration.

Web search can be added later as a tool.

It belongs under future improvements rather than becoming a source of unnecessary scope expansion.

---

# 44. Why Four Agents?

The four agents represent distinct responsibilities:

```text
Researcher
Find information.

Analyst
Understand and organize information.

Writer
Create the report.

Reviewer
Check the result.
```

This makes the multi-agent architecture easy to explain and test.

---

# 45. Why Not Make Each Agent an LLM?

Because agent architecture and LLM usage are different concepts.

An agent can contain:

- deterministic logic;
- tools;
- routing;
- retrieval;
- an LLM;
- validation.

The project deliberately demonstrates that not every agent needs a separate LLM generation.

This improves:

- latency;
- predictability;
- testability;
- local CPU practicality.

---

# 46. Interview Preparation

## Q1. What is Advanced RAG?

Advanced RAG improves basic retrieval using techniques such as:

- hybrid search;
- query rewriting;
- metadata filtering;
- reranking;
- better candidate selection.

## Q2. Why use hybrid retrieval?

Vector retrieval handles semantic similarity.

BM25 handles lexical similarity and exact terms.

Hybrid retrieval combines both to improve retrieval robustness.

## Q3. What is Reciprocal Rank Fusion?

RRF combines rankings from multiple retrieval systems.

Instead of directly comparing incompatible raw scores, it rewards documents that appear highly ranked across multiple retrieval methods.

## Q4. Why use BM25 when we already have embeddings?

Embeddings can miss exact terminology.

BM25 is strong for:

- identifiers;
- technical terms;
- exact phrases;
- keyword-heavy queries.

Therefore BM25 complements semantic retrieval.

## Q5. What is reranking?

Reranking applies a stronger relevance model to the candidates returned by initial retrieval.

The CrossEncoder evaluates the query and document together.

## Q6. Why rerank only a small candidate set?

CrossEncoder inference is more expensive than initial retrieval.

Therefore:

```text
Fast retrieval
    ↓
Top-K candidates
    ↓
Expensive reranking
```

is more scalable than reranking the entire corpus.

## Q7. What is query rewriting?

Query rewriting transforms a user query into another formulation that may be more suitable for retrieval.

## Q8. What can go wrong with query rewriting?

A generative model may change the user's intent.

This is called semantic drift.

The solution is to validate the rewrite and fall back to the original query when necessary.

## Q9. What is an agent?

An agent is a system component that can use tools, state, reasoning or workflow logic to accomplish a task rather than simply performing one fixed function.

## Q10. Why use multiple agents?

Different agents can specialize in different responsibilities.

In this project:

```text
Researcher → evidence
Analyst    → synthesis
Writer     → report
Reviewer   → quality
```

## Q11. Why LangGraph?

LangGraph provides explicit stateful orchestration.

It is particularly useful for:

- state;
- nodes;
- edges;
- conditional routing;
- iterative workflows;
- agent coordination.

## Q12. What is LangGraph state?

State is the shared information carried between workflow nodes.

For this project it includes information such as:

```text
question
retrieval_query
evidence
analysis
draft
review
approved
```

## Q13. What is conditional routing?

Conditional routing allows the graph to choose the next node based on current state.

Example:

```text
Reviewer
   |
   +---- approved → END
   |
   +---- rejected → revision
```

## Q14. Why separate Researcher and Analyst?

The Researcher focuses on finding evidence.

The Analyst focuses on interpreting and organizing that evidence.

Separating responsibilities makes the workflow easier to test and extend.

## Q15. How do you handle failures?

Examples:

- invalid questions raise validation errors;
- empty evidence causes rejection;
- empty drafts are rejected;
- invalid query rewrites fall back to the original query;
- reviewer checks prevent unsupported/structurally invalid reports from being approved.

## Q16. What is the biggest performance limitation?

Local CPU inference.

Qwen query rewriting takes several seconds, so unnecessary LLM generations are avoided.

## Q17. Why use a small model?

The project is intentionally:

```text
Local
Zero-cost
CPU-compatible
```

A larger model would increase latency substantially without adding enough project value for the target concepts.

## Q18. What would you improve in production?

Potential improvements include:

- stronger query-rewriting models;
- GPU inference;
- quantization;
- persistent vector databases;
- web-search tools;
- MCP;
- better observability;
- semantic evaluation;
- LLM-based review;
- authentication;
- API deployment;
- distributed execution.

These are intentionally outside the current Project 7 scope.

---

# 47. Limitations

The current system has several deliberate limitations.

### Model

Qwen2.5-0.5B is small and can produce imperfect rewrites.

### Hardware

CPU inference is slower than GPU inference.

### Corpus

The corpus is small and local.

### Reviewer

The reviewer performs deterministic checks rather than complete semantic evaluation.

### Retrieval

Hybrid retrieval and reranking improve relevance but do not guarantee factual correctness.

### Query Rewriting

Generated rewrites can introduce semantic drift.

### Persistence

The current research workflow does not provide a persistent research database.

### External Tools

No external web-search or MCP tool integration is included.

### Production

No production authentication, multi-user isolation, distributed execution, or deployment infrastructure is included.

---

# 48. Future Improvements

The following are intentionally postponed:

## Retrieval

- stronger embedding models;
- larger rerankers;
- hybrid weighting experiments;
- persistent vector databases;
- larger corpora.

## Agents

- LLM-powered Analyst;
- LLM-powered Writer;
- LLM-powered Reviewer;
- richer planning;
- tool selection.

## Tools

- web search;
- browser tools;
- MCP tools;
- document connectors.

## Evaluation

- semantic evaluation;
- LLM-as-judge;
- retrieval metrics;
- groundedness metrics;
- answer relevance metrics.

## Production

- FastAPI;
- Docker;
- observability;
- tracing;
- token tracking;
- latency monitoring;
- authentication;
- deployment.

These improvements are deliberately postponed to prevent Project 7 scope creep.

---

# 49. Project 7 Completion Checklist

## Advanced RAG

- [x] Vector retrieval
- [x] BM25 retrieval
- [x] Hybrid retrieval
- [x] Reciprocal Rank Fusion
- [x] Metadata filtering
- [x] Query rewriting
- [x] Query rewrite fallback
- [x] CrossEncoder reranking

## Agents

- [x] Researcher
- [x] Analyst
- [x] Writer
- [x] Reviewer

## LangGraph

- [x] State
- [x] Nodes
- [x] Edges
- [x] Conditional routing
- [x] Workflow orchestration

## Testing

- [x] Unit tests
- [x] Integration tests
- [x] Failure tests
- [x] End-to-end workflow test
- [x] Real local model validation

## Engineering

- [x] UV environment
- [x] Python 3.12
- [x] Local model cache reuse
- [x] `.gitignore`
- [x] `uv.lock`
- [x] Scratch-file discipline
- [x] Failure analysis
- [x] Performance measurements
- [x] Limitations documented

## Documentation

- [x] README
- [x] Architecture
- [x] Technology stack
- [x] Installation
- [x] Testing
- [x] Performance
- [x] Limitations
- [x] Failure analysis
- [x] Interview preparation
- [x] Future improvements

## Release

- [x] Core implementation complete
- [x] End-to-end workflow validated
- [x] 25 tests passing
- [ ] Final Git commit
- [ ] Final Git push
- [ ] Final clean working tree verification

---

# 50. Project Status

## Current Status

**Project 7 — Advanced Agentic RAG & Multi-Agent Research System**

Core implementation:

**COMPLETE**

Automated tests:

```text
25 passed
```

End-to-end workflow:

**VALIDATED**

Advanced RAG:

**COMPLETE**

Multi-agent architecture:

**COMPLETE**

LangGraph workflow:

**COMPLETE**

Failure handling:

**COMPLETE**

Documentation:

**COMPLETE**

Final Git release:

**Pending final commit/push verification**

---

# 51. Learning Outcome

After completing this project, the key progression is:

```text
Project 1
AI Text Generation
        ↓
Project 2
Basic RAG
        ↓
Project 3
Tool Calling
        ↓
Project 4
Structured Output
        ↓
Project 5
Evaluation & Guardrails
        ↓
Project 6
Conversation Memory & Context
        ↓
Project 7
Advanced RAG + Agents + LangGraph
        ↓
Project 8
Production-Style GenAI Application
```

Project 7 therefore serves as the bridge from individual GenAI capabilities to **stateful, retrieval-driven, multi-agent systems**.

---

# 52. Final Engineering Principle

The project follows these engineering principles:

```text
Reuse before rebuilding
Measure before optimizing
Deterministic before unnecessary LLM calls
Isolate expensive model inference
Test normal and failure paths
Keep experiments temporary
Document limitations honestly
Record failures and prevention rules
Commit at logical milestones
Protect project scope
```

The objective is not maximum production complexity.

The objective is:

> **Build → Understand → Explain → Test → Document → Analyze → Release**
