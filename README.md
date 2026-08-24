# Document Indexing & Retrieval

A Python module that extracts text from PDF/DOCX files, chunks it three
different ways, embeds it with `gemini-embedding-001`, stores it in
PostgreSQL via pgvector, and supports semantic search over the result.

Built for the Jeen.ai AI Solutions Engineer home assignment (Part 2).

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

Start Postgres with the pgvector extension and schema already applied:

```bash
docker compose up -d
```

## Environment variables

Set in `.env` (see `.env.example`):

```ini
GEMINI_API_KEY=your-key-here
POSTGRES_URL=postgresql://jeen:jeen_dev_password@localhost:5433/document_index
```

- `GEMINI_API_KEY` — free at https://aistudio.google.com/apikey
- `POSTGRES_URL` — matches `docker-compose.yml`'s default (port 5433, not
  5432, to avoid colliding with any Postgres already running locally)

## Run examples

**Indexing:**

```bash
python index_documents.py --file ./docs/sample_tariff_guide.pdf --strategy paragraph
python index_documents.py --file ./docs/sample_support_procedures.docx --strategy sentence
python index_documents.py --file ./docs/sample_support_procedures.docx --strategy fixed --chunk-size 800 --overlap 100
```

`--strategy` is one of `fixed`, `sentence`, `paragraph`. `--chunk-size` and
`--overlap` (characters, not tokens) are optional and default to 1000/150.

**Searching:**

```bash
python search.py --query "login issue"
python search.py --query "login issue" --limit 3
```

## Sample output

Two synthetic sample documents are included under `docs/`:
`sample_tariff_guide.pdf` and `sample_support_procedures.docx`.

**Indexing run:**

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

**Search run:**

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

**Search results returned directly from the database:**

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
