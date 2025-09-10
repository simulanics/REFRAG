
# REFRAG Test Corpora

Generated on 2025-09-09 for synthetic testing of the REFRAG pipeline.
All content is synthetic/templated for testing & debugging with deterministic values.

Files:
- `corpus_small.txt`  — 500 lines (1 passage per line)
- `corpus_medium.txt` — 2,000 lines
- `corpus_large.txt`  — 3,000 lines
- `rag_train.jsonl`   — 1,200 QA pairs aligned to deterministic facts embedded in the corpora
- `cpt_train.jsonl`   — 400 long-text items for continual pretraining (reconstruction & next-paragraph)

## Usage

**Build an index** (example using the large corpus):

```bash
python refrag.py index --corpus data/corpus_large.txt --index_dir runs/index_large --embed_model BAAI/bge-small-en-v1.5
```

**Train policy** using the synthetic QA:

```bash
python refrag.py train_policy --rag_json data/rag_train.jsonl --index_dir runs/index_large --topk 8 --k 64 --p 0.25
```

**Generate**:

```bash
python refrag.py generate --index_dir runs/index_large --question "Which river flows through City_101?" --topk 8 --k 64 --p 0.25
```

Ground-truth is embedded in text (e.g., for `City_101`, river is deterministically `River_...`).

## Notes

- The text is synthetic and multilingual sprinkles are included to test tokenization (English/Spanish/Chinese).
- Q/A values are deterministic functions of numeric IDs so correctness can be verified without external sources.
- For performance tests, prefer `corpus_medium.txt` or `corpus_large.txt`. For CI, use `corpus_small.txt`.
