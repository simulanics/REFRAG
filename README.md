# REFRAG-style RAG (compress → sense/select → expand)

A compact, complete reference implementation of a **REFRAG-style retrieval-augmented generation** pipeline:
- **Compress** long contexts into **per-chunk embeddings** with an encoder.
- **Project** those chunk vectors into the decoder’s token-embedding space.
- **Selectively expand** the most informative chunks back to **full token embeddings** (policy / heuristic).
- **Decode** while measuring TTFT/TTIT/throughput.

This repo includes:
- `refrag.py` — single-file implementation (retrieval, encoder/projector, selective expansion, CPT, generation).
- Auto-accelerated quickstarts:
  - `refrag_quickstart_auto_accel.sh` (Linux/macOS) — detects **CUDA → ROCm → MPS → CPU**
  - `refrag_quickstart_auto_accel.bat` (Windows) — detects **CUDA → CPU**

> **Paper basis:** “RETHINKING RAG based Decoding (REFRAG)” — this re-creates the **compress → sense/select → expand** architecture described in the first 11 pages of the paper.

---

## Features

- 🔎 **Retrieval** with FAISS (index build and search)
- 🧱 **Chunk encoder** (CLS pooling) + **token-space projector**
- 🎯 **Selective expansion** via a tiny **policy network** (REINFORCE) with a strong **PPL heuristic** fallback
- 📚 **Continual pretraining** (CPT) curricula: **reconstruction → next-paragraph prediction**
- 🧪 **Generation metrics**: TTFT, TTIT, throughput
- 🧰 Single **CLI** with subcommands

---

## Installation & Acceleration Matrix

| OS / HW | PyTorch install | FAISS | Notes |
|---|---|---|---|
| **Linux + NVIDIA CUDA** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` | Try `faiss-gpu`, fallback `faiss-cpu` | CUDA 12.1 wheels. |
| **Linux + AMD ROCm** | `pip install --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision torchaudio` | `faiss-cpu` | ROCm 6.x. FAISS GPU via pip is not available; use CPU or build from source for GPU. |
| **macOS (Apple Silicon/Intel)** | `pip install torch torchvision torchaudio` | `faiss-cpu` | Apple Silicon supports **MPS**. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`. |
| **Windows (NVIDIA/CPU)** | CUDA: same cu121 line above; otherwise CPU: `--index-url https://download.pytorch.org/whl/cpu` | `faiss-cpu` | `faiss-gpu` wheels are not provided on pip for Windows. |

> The provided scripts auto-detect your accelerator and install the correct wheels. They also patch `refrag.py` so `now_device()` prefers **CUDA → MPS → CPU** (ROCm appears as `torch.cuda.is_available()` in PyTorch).

---

## Quickstart (scripts)

> Put `refrag.py` in the repo root (same folder as the scripts).

### Linux/macOS (auto CUDA/ROCm/MPS/CPU)
```bash
chmod +x refrag_quickstart_auto_accel.sh
./refrag_quickstart_auto_accel.sh
```

### Windows (auto CUDA/CPU)
```bat
refrag_quickstart_auto_accel.bat
```

Environment variables (optional overrides): `ENC_MODEL`, `DEC_MODEL`, `EMBED_MODEL`, `TOPK`, `K`, `P`, `CTX_MAX`, `MAX_NEW`, `STEPS`, `LR_RECON`, `LR_NEXT`, `LR_POLICY`.

---

## Manual Usage (CLI)

### 0) Create a venv and install deps
Pick the commands for your platform from the table above (CUDA/ROCm/MPS/CPU), plus:
```bash
pip install "transformers==4.43.3" accelerate sentencepiece sacrebleu numpy faiss-cpu
# (Linux+CUDA users can try: pip install faiss-gpu)
```

### 1) Build a FAISS index
```bash
python refrag.py index   --corpus data/wiki_lines.txt \  # one passage per line
  --index_dir runs/index   --embed_model BAAI/bge-small-en-v1.5
```

### 2) Generate (RAG with compression/expansion)
```bash
python refrag.py generate   --index_dir runs/index   --embed_model BAAI/bge-small-en-v1.5   --enc roberta-base   --dec meta-llama/Llama-3.2-3B   --question "Who discovered penicillin?"   --topk 4   --k 32   --p 0.25   --ctx_max 1024   --max_new 128   --temperature 0.0
# Add --heuristic to bypass RL policy and use PPL-based selection.
```

### 3) Continual Pretraining (CPT)

**Phase A — Reconstruction (freeze decoder):**
```bash
python refrag.py cpt_recon   --train_json data/cpt_train.jsonl   --enc roberta-base   --dec meta-llama/Llama-3.2-3B   --k 64   --steps 300   --lr 2e-5   --log_every 20   --out_dir runs/cpt_recon
```

**Phase B — Next-paragraph prediction (unfreeze all):**
```bash
python refrag.py cpt_next   --train_json data/cpt_train.jsonl   --enc roberta-base   --dec meta-llama/Llama-3.2-3B   --k 64   --steps 300   --lr 2e-5   --expand_frac 0.25   --log_every 20   --load_dir runs/cpt_recon   --out_dir runs/cpt_next
```

### 4) Train the Selective-Expansion Policy (REINFORCE)
```bash
python refrag.py train_policy   --rag_json data/rag_train.jsonl   --index_dir runs/index   --embed_model BAAI/bge-small-en-v1.5   --enc roberta-base   --dec meta-llama/Llama-3.2-3B   --k 32   --steps 300   --lr 1e-4   --p 0.25   --topk 4   --log_every 20   --out_dir runs/policy
```

### 5) Generate with Saved Weights
```bash
python refrag.py generate   --index_dir runs/index   --embed_model BAAI/bge-small-en-v1.5   --enc roberta-base   --dec meta-llama/Llama-3.2-3B   --load_dir runs/cpt_next \   # or runs/policy
  --question "Explain how penicillin was discovered and by whom."   --topk 4 --k 32 --p 0.25 --max_new 192
```

---

## FAISS Notes

- **CUDA (Linux + NVIDIA):** The script attempts `faiss-gpu`. If pip fails (no wheel), it falls back to `faiss-cpu`.
- **ROCm (Linux + AMD):** Use `faiss-cpu`. GPU FAISS wheels for ROCm are not provided on PyPI; build from source if you need GPU FAISS.
- **macOS & Windows:** Use `faiss-cpu`. (Windows has no official `faiss-gpu` wheel on pip.)

---

## Troubleshooting

- **Gated Hugging Face models** → `huggingface-cli login` and accept the model license on the Hub.
- **CUDA OOM** → use a smaller decoder, lower `--ctx_max`, `--k`, `--max_new`, or reduce `--p`.
- **MPS quirks** → set `PYTORCH_ENABLE_MPS_FALLBACK=1` (already in the script). Some ops may run on CPU fallback.
- **ROCm install** → ensure ROCm runtime is installed (`rocminfo` should work). Wheels: `--index-url https://download.pytorch.org/whl/rocm6.0`.
- **FAISS build from source** (optional for ROCm GPU): see FAISS docs; otherwise default to `faiss-cpu`.

---

## Contributing

We welcome issues and PRs. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** and abide by our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## License

This project is licensed under the **MIT License** — see **[LICENSE](LICENSE)**.


## Demo: Synthetic Test Corpus

What’s inside (`refrag/data/`):

* `corpus_small.txt` — 500 passages (1 per line)
* `corpus_medium.txt` — 2,000 passages
* `corpus_large.txt` — 3,000 passages
* `rag_train.jsonl` — 1,200 synthetic QA pairs aligned to the corpus (answers are deterministically embedded in docs)
* `cpt_train.jsonl` — 400 long-form items for continual pretraining (reconstruction & next-paragraph)
* `README_DATA.md` — usage, tips, and examples
* `make_corpus.py` — a tiny reproducibility stub

### Quick usage

**Build an index** (example with the large corpus):

```bash
python refrag.py index --corpus data/corpus_large.txt --index_dir runs/index_large --embed_model BAAI/bge-small-en-v1.5
```

**Train policy** on synthetic QA:

```bash
python refrag.py train_policy --rag_json data/rag_train.jsonl --index_dir runs/index_large --topk 8 --k 64 --p 0.25
```

**Generate**:

```bash
python refrag.py generate --index_dir runs/index_large --question "Which river flows through City_101?" --topk 8 --k 64 --p 0.25
```

**Notes**

* Corpus spans four templates (cities, alloys, biographies, events) with multilingual sprinkles (EN/ES/ZH) to stress tokenization.
* QA ground truth (e.g., `River_<id>`, `University_<id>`) is deterministic, so you can automatically validate retrieval and answers.
* For CI or smoke tests use `corpus_small.txt`; for perf, use `corpus_medium.txt`/`corpus_large.txt`. If you want a **10k+** mega set, say the word and we’ll spin one up the same way.


## CLI Quick Reference (Updated)

`refrag.py` exposes subcommands to build an index, run continual pretraining, train the selective-expansion policy, and generate answers.

### Device Selection
Runs on **CUDA** (incl. ROCm builds), **Apple MPS**, then **CPU** automatically.  
Install the appropriate PyTorch/FAISS wheels:
- NVIDIA CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` and `pip install faiss-gpu` (or `faiss-cpu` if needed)
- AMD ROCm (Linux): install ROCm PyTorch per PyTorch docs and `faiss-gpu` if available for your ROCm stack
- Apple Silicon (MPS): stock CPU wheels typically include MPS; use `faiss-cpu`
- Generic CPU: `pip install torch torchvision torchaudio`, `pip install faiss-cpu`

### Commands

#### 1) Build FAISS index
```bash
python refrag.py index --corpus data/corpus_large.txt --index_dir runs/index_large --embed_model BAAI/bge-small-en-v1.5
```
- `--corpus`: text file, one passage per line
- `--index_dir`: output directory containing `texts.npy` and `faiss.index`

#### 2) Continual pretraining — Reconstruction (CPT-A)
```bash
python refrag.py cpt_recon --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-3B --k 64 --steps 1000
```

#### 3) Continual pretraining — Next paragraph (CPT-B)
```bash
python refrag.py cpt_next --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-3B --k 64 --steps 1000 --expand_frac 0.25 --load_dir runs/cpt_recon
```

#### 4) Train selective-expansion policy (REINFORCE)
```bash
python refrag.py train_policy --rag_json data/rag_train.jsonl --index_dir runs/index_large --topk 8 --k 64 --p 0.25
```

#### 5) Generate answers
```bash
python refrag.py generate --index_dir runs/index_large --question "Which river flows through City_101?" --topk 8 --k 64 --p 0.25
```
- `--heuristic` flag switches to heuristic expansion instead of learned policy.
- `--load_dir` can point to saved weights: `encoder.pt`, `projector.pt`, `policy.pt`, or `refrag_full.pt`.

