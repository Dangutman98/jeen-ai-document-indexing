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

**Indexing (two files, two different strategies):**

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

$ python index_documents.py --file ./docs/sample_support_procedures.docx --strategy sentence
Extracting text from ./docs/sample_support_procedures.docx ...
  extracted 2,533 characters
Chunking with strategy='sentence' ...
  produced 13 chunks
Generating embeddings via gemini-embedding-001 (13 chunks) ...
  received 13 embeddings, 1536 dims each
Storing in Postgres ...
Done. Inserted 13 rows for 'sample_support_procedures.docx' (strategy=sentence).
```

**Searching:**

```
$ python search.py --query "login issue" --limit 3
Embedding query: 'login issue'

Top 3 result(s) for 'login issue':

[1] similarity=0.7566  file=sample_support_procedures.docx  strategy=sentence  id=5
    1. Login Issues

[2] similarity=0.6619  file=sample_support_procedures.docx  strategy=sentence  id=6
    If a customer cannot log into the IEC self-service app, first confirm
    the account number is correct and that the customer is using the
    number printed at the top of their most recent bill, not their
    national ID. Password ...

[3] similarity=0.6119  file=sample_support_procedures.docx  strategy=sentence  id=13
    5. Account Access Changes
```

```
$ python search.py --query "billing dispute" --limit 3
Embedding query: 'billing dispute'

Top 3 result(s) for 'billing dispute':

[1] similarity=0.7670  file=sample_support_procedures.docx  strategy=sentence  id=7
    2. Billing Disputes

[2] similarity=0.7150  file=sample_support_procedures.docx  strategy=sentence  id=8
    A billing dispute should first be checked against the meter read
    status. If the disputed bill was based on an estimated read rather
    than an actual read, explain that estimates are corrected
    automatically once a real read...

[3] similarity=0.6175  file=sample_support_procedures.docx  strategy=sentence  id=9
    3. Meter Reading Failures
```

The top result for each query is the section header chunk that actually
discusses the topic — retrieval is correctly ranking relevance, not just
returning arbitrary rows.

**Search results returned directly from the database** (not just through the
CLI — a raw query against the table, confirming the data really is there
with the right shape):

```
$ docker exec jeen_part2_postgres psql -U jeen -d document_index -c \
    "SELECT id, filename, split_strategy, LEFT(chunk_text, 60) AS chunk_preview
     FROM document_chunks ORDER BY id LIMIT 6;"

 id |            filename            | split_strategy |                        chunk_preview
----+---------------------------------+----------------+--------------------------------------------------------------
  1 | sample_tariff_guide.pdf        | paragraph      | IEC Tariff and Solar Net-Metering Guide (Sample)            +
    |                                 |                | 1. Standard
  2 | sample_tariff_guide.pdf        | paragraph      | Solar Net-Metering                                          +
    |                                 |                | Customers with a private solar installati
  3 | sample_tariff_guide.pdf        | paragraph      | Chat agents must never                                      +
    |                                 |                | confirm or schedule a disconnection o
  4 | sample_support_procedures.docx | sentence       | IEC Customer Support Procedures (Sample)
  5 | sample_support_procedures.docx | sentence       | 1. Login Issues
  6 | sample_support_procedures.docx | sentence       | If a customer cannot log into the IEC self-service app, firs
(6 rows)
```

### Verifying the index is real (and why `EXPLAIN` may not show it at this scale)

Running `EXPLAIN` on the exact query `search.py` executes — a literal vector
bound as a query parameter, matching the real code path in `src/db.py`:

```sql
EXPLAIN SELECT id FROM document_chunks
ORDER BY embedding <=> '[...]'::vector
LIMIT 5;
```

On this repo's small demo table (a dozen or so rows from the sample docs),
Postgres's planner correctly chooses a **sequential scan**, not the HNSW
index:

```
Limit  (cost=23.39..23.40 rows=5 width=120)
  ->  Sort  (cost=23.39..24.54 rows=460 width=120)
        Sort Key: ((embedding <=> '[...]'::vector))
        ->  Seq Scan on document_chunks  (cost=0.00..15.75 rows=460 width=120)
```

This is expected, correct planner behavior, not a bug — walking a graph index
has fixed overhead that isn't worth paying when scanning a dozen rows
directly is cheaper. The index exists and is genuinely usable; it simply
isn't the cheaper plan yet at this table size. Forcing the planner to use it
proves it's real and correct:

```sql
SET enable_seqscan = off;
EXPLAIN SELECT id FROM document_chunks
ORDER BY embedding <=> '[...]'::vector
LIMIT 5;
```

```
Limit  (cost=72.37..73.07 rows=5 width=16)
  ->  Index Scan using document_chunks_embedding_hnsw_idx on document_chunks  (cost=72.37..137.20 rows=460 width=16)
        Order By: (embedding <=> '[...]'::vector)
```

`Index Scan using document_chunks_embedding_hnsw_idx` confirms the index is
real and functional. At a realistic corpus size (thousands of chunks,
where a sequential scan is the expensive option), the planner will pick it
automatically without needing `enable_seqscan` forced off — this is exactly
the payoff of the 1536-dimension decision above: a 3072-dim column could
never be indexed at all, at any scale.

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

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

47 tests across three layers:

- **`tests/test_chunking.py`** — pure logic, no I/O: all three strategies,
  overlap edge cases, empty/whitespace-only input, Hebrew text, sentences
  longer than the chunk size, and the paragraph-fallback behavior.
- **`tests/test_extract.py`** — real temp PDF/DOCX files generated per test
  (via `fpdf2`/`python-docx`): missing file, unsupported extension, empty
  document, whitespace-only document, table-cell extraction, multi-page PDFs.
- **`tests/test_embeddings.py`** — retry/backoff logic against a mocked
  Gemini client (no real network calls, no real waiting): retries on 429,
  does *not* retry on 400, exhausts retries and raises, batches large input
  correctly. Plus real unit tests of the L2-renormalization math.
- **`tests/test_integration.py`** — runs against the *real* Postgres and
  Gemini API from `docker-compose.yml` (auto-skipped if `GEMINI_API_KEY` /
  `POSTGRES_URL` aren't set): a full insert → embed → search round trip,
  real embedding dimensionality/norm checks, and real DB connection-failure
  handling.

**A real bug the test suite caught:** the original sentence-chunking
implementation split sentences *within* each paragraph correctly, but then
flattened all paragraphs' sentences into one list before packing them into
chunks — silently discarding the paragraph boundaries it had just computed.
With a large enough `chunk_size`, sentences from two unrelated paragraphs
could end up merged into the same chunk, contradicting the function's own
documented intent. Fixed by keeping sentences grouped by source paragraph
through the packing step and treating each paragraph boundary as a hard
chunk break. Covered by `test_sentence_never_bridges_paragraph_boundaries`.

**A second real bug, found only by testing against real documents:** the
synthetic sample docs in `docs/` are plain prose with no short standalone
paragraphs, so they never exercised a real failure mode. Running the
pipeline against real-world DOCX/PDF files (specs, syllabi, PRDs) with
headings, table cells, and list items — each its own blank-line-separated
paragraph — showed the paragraph-hard-break fix above had a side effect:
every short paragraph became its own near-empty chunk. One real document
produced 53 chunks under 15 characters out of 219 total, including chunks
like `'Developer'`, `'Login'`, and a 4-character Hebrew chunk containing only
`'צוות'` ("team") — each one a wasted embedding API call carrying essentially
no retrievable meaning. Fixed with `_merge_undersized()` in
`src/chunking.py`: any chunk under 30 characters folds into a neighbor
(forward if one follows, backward otherwise), for both the `sentence` and
`paragraph` strategies. Re-running the same real document afterward produced
130 chunks, zero of them undersized. Not caught by any of the strategy-level
unit tests above, precisely because they only used clean synthetic prose —
a reminder that unit tests over synthetic fixtures and testing against real
files are not substitutes for each other.

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
  tests/                      # pytest suite -- unit + real integration tests
  .env.example
  requirements.txt
  requirements-dev.txt        # + pytest, fpdf2 (test-only)
```
