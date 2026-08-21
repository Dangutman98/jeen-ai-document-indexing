# Document Indexing & Retrieval

A Python module that extracts text from PDF/DOCX files, chunks it three
different ways, embeds it with `gemini-embedding-001`, stores it in
PostgreSQL via pgvector, and supports semantic search over the result.

Built for the Jeen.ai AI Solutions Engineer home assignment (Part 2).

## A note on the embedding dimensions

`gemini-embedding-001` returns **3072-dimension** vectors by default. pgvector's
HNSW and IVFFlat indexes only support up to **2000 dimensions** — a 3072-dim
column can be stored, but it can never be indexed, so every similarity query
silently falls back to a full sequential scan regardless of table size.

This project requests `output_dimensionality=1536` from the API (Matryoshka
truncation — one of Google's own recommended output sizes, alongside 3072 and
768) and L2-renormalizes the returned vector, which Google's docs require for
any non-default output size since the API does not renormalize truncated
vectors itself. See [`src/embeddings.py`](src/embeddings.py). This lets the
table carry a real HNSW index — confirmed in this repo with `EXPLAIN`, see
[Verifying the index is actually used](#verifying-the-index-is-actually-used)
below.

## Requirements

- Python 3.11+
- Docker (for Postgres + pgvector)
- A Gemini API key — free at https://aistudio.google.com/apikey

## Installation

```bash
cd part2
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Copy the env template and fill in your key:

```bash
cp .env.example .env
```

```ini
GEMINI_API_KEY=your-key-here
POSTGRES_URL=postgresql://jeen:jeen_dev_password@localhost:5433/document_index
```

> **Port 5433, not 5432** — `docker-compose.yml` maps the container to host
> port 5433 to avoid colliding with any Postgres instance you might already
> have running locally on the default port. Change it in both
> `docker-compose.yml` and `.env` if 5433 is also taken on your machine.

Start Postgres with the pgvector extension and schema already applied:

```bash
docker compose up -d
```

The schema (`schema.sql`) is applied automatically on first container start
via Postgres's `docker-entrypoint-initdb.d` mechanism. It creates the
`document_chunks` table and the HNSW index described above.

## Usage

### Indexing a document

```bash
python index_documents.py --file ./docs/example.pdf --strategy paragraph
python index_documents.py --file ./docs/example.docx --strategy sentence
python index_documents.py --file ./docs/example.docx --strategy fixed --chunk-size 800 --overlap 100
```

`--strategy` is one of `fixed`, `sentence`, `paragraph`. `--chunk-size` and
`--overlap` (characters, not tokens) are optional and default to 1000/150.

### Searching

```bash
python search.py --query "login issue"
python search.py --query "login issue" --limit 3
```

## Example run (real output from this repo)

Two synthetic sample documents are included under `docs/` so the pipeline is
testable without sourcing external files: `sample_tariff_guide.pdf` and
`sample_support_procedures.docx`, both fictional IEC-style support content.

**Indexing:**

```
$ python index_documents.py --file ./docs/sample_tariff_guide.pdf --strategy paragraph
Extracting text from ./docs/sample_tariff_guide.pdf ...
  extracted 1,959 characters
Chunking with strategy='paragraph' ...
  produced 3 chunks
Generating embeddings via gemini-embedding-001 (3 chunks) ...
  received 3 embeddings, 1536 dims each
Storing in Postgres ...
Done. Inserted 3 rows for 'sample_tariff_guide.pdf' (strategy=paragraph).
```

**Searching:**

```
$ python search.py --query "login issue" --limit 3
Embedding query: 'login issue'

Top 3 result(s) for 'login issue':

[1] similarity=0.6178  file=sample_support_procedures.docx  strategy=sentence  id=13
    IEC Customer Support Procedures (Sample) 1. Login Issues If a customer
    cannot log into the IEC self-service app, first confirm the account
    number is correct and that the customer is using the number printed
    at the top of...

[2] similarity=0.5913  file=sample_support_procedures.docx  strategy=fixed  id=16
    IEC Customer Support Procedures (Sample)  1. Login Issues  If a
    customer cannot log into the IEC self-service app, first confirm the
    account number is correct and that the customer is using the number
    printed at the top ...

[3] similarity=0.5880  file=sample_support_procedures.docx  strategy=sentence  id=14
    When a meter has not been read for two or more consecutive billing
    periods, this is classified as a meter reading failure. The customer
    should be offered a self-reading submission through the app as an
    interim fix, and a...
```

The top result is the chunk that actually discusses login — retrieval is
working correctly across the `sentence`, `fixed`, and `paragraph` strategies
in the same table.

### Verifying the index is actually used

```sql
EXPLAIN SELECT id FROM document_chunks
ORDER BY embedding <=> (SELECT embedding FROM document_chunks LIMIT 1)
LIMIT 5;
```

```
Limit  (cost=10000000034.42..10000000035.01 rows=5 width=16)
  InitPlan 1
    ->  Limit  (cost=10000000000.00..10000000000.03 rows=1 width=32)
          ->  Seq Scan on document_chunks document_chunks_1  ...
  ->  Index Scan using document_chunks_embedding_hnsw_idx on document_chunks  (cost=34.38..89.20 rows=460 width=16)
        Order By: (embedding <=> (InitPlan 1).col1)
```

`Index Scan using document_chunks_embedding_hnsw_idx` confirms the HNSW index
is doing the work, not a full table scan — this is the payoff of the
1536-dimension decision above.

## Error handling

All six mandatory cases are handled with a specific exception type
(`src/errors.py`) and a clear message, verified against real conditions:

| Case | Behavior |
|---|---|
| Missing file | `FileNotFoundPipelineError` before any extraction is attempted |
| Unsupported file type | `UnsupportedFileTypeError`, lists supported extensions |
| Document with no extractable text | `NoExtractableTextError` (e.g. a scanned/image-only PDF, or an empty DOCX) |
| Embedding failure | `EmbeddingError` — missing API key fails fast; transient API errors (429/5xx) retry with exponential backoff before failing |
| Database connection failure | `DatabaseConnectionError` with a hint to check `docker compose up` |
| Empty search results | `EmptySearchResultError` when the table has zero rows |

**A note on "empty search results":** pgvector's `ORDER BY ... LIMIT` always
returns the *N nearest* rows by distance, however irrelevant — there is no
built-in similarity threshold. So this error case fires specifically when the
table itself has zero rows (nothing has been indexed yet), not when a query
merely has no strong match. This is standard vector-search behavior, not a
gap in this implementation.

## Design notes

- **Modules are single-purpose** (`extract.py`, `chunking.py`,
  `embeddings.py`, `db.py`) so each can be read, tested, and changed in
  isolation.
- **No hardcoded paths.** Everything flows through CLI arguments or `.env`.
- **No secrets are ever printed.** Error messages reference environment
  variable names, never their values.
- **Embeddings are batched** (32 chunks/request) with retry/backoff, since
  the free tier is rate-limited and a real document can produce far more
  chunks than a naive one-request-per-chunk approach could sustain.
- **Paragraph chunking has a fallback.** Some formats (notably PDF text
  layers extracted via `pypdf`) carry no blank-line paragraph markers at
  all — the whole page comes back as one block. Rather than silently return
  the entire document as a single chunk, `paragraph` strategy falls back to
  sentence-grouping when no paragraph structure is detected.
- **Retrieval task types are asymmetric.** Documents are embedded with
  `task_type="RETRIEVAL_DOCUMENT"` and queries with `RETRIEVAL_QUERY"`,
  per Google's guidance for retrieval-quality embeddings — using the same
  task type for both would silently degrade match quality.

## Project structure

```
part2/
  docker-compose.yml        # pgvector/pgvector:pg17, host port 5433
  schema.sql                 # table + HNSW index, applied on first container start
  index_documents.py         # CLI entry point
  search.py                  # CLI entry point
  docs/                      # synthetic sample PDF/DOCX for testing
  src/
    extract.py                # PDF/DOCX -> clean text
    chunking.py                # fixed / sentence / paragraph strategies
    embeddings.py               # Gemini client, batching, retry, dim reduction
    db.py                        # pgvector storage + similarity search
    errors.py                     # typed exceptions for all 6 error cases
  .env.example
  requirements.txt
```
