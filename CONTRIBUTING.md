# Contributing

Thanks for your interest in improving this project!

## Ways to contribute
- Bug reports and reproduction cases
- Docs improvements (README examples, platform notes)
- Performance tweaks (better chunking, projector variants, caching strategies)
- New retrieval/selection strategies (e.g., different policies, heuristics)
- Tests (unit/smoke) and CI

## Development setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Choose your platform install (CUDA/ROCm/MPS/CPU) then:
pip install "transformers==4.43.3" accelerate sentencepiece sacrebleu numpy faiss-cpu
```

## Style
- Python ≥ 3.10, follow **PEP 8**.
- Use **type hints** where practical.
- Keep functions focused and documented with short docstrings.
- Prefer explicitness over magic.

## Commit messages
Use Conventional Commits where possible:
```
feat: add ROCm detection
fix: handle empty corpus gracefully
docs: expand README on FAISS choices
perf: cache chunk encodings across steps
refactor: split projector into MLP and linear
test: add smoke test for generate
```
Link issues with `Fixes #123` when appropriate.

## Pull requests
- Create a topic branch; keep PRs focused and reasonably small.
- Include a clear description and any platform-specific notes (CUDA/ROCm/MPS).
- If your PR changes defaults or interfaces, update the README and examples.

## Running quick smoke tests
- Build a tiny index (`data/wiki_lines.txt`) and run `generate` end-to-end.
- On CUDA boxes, ensure `faiss-gpu` is attempted; on ROCm/macOS/Windows, confirm `faiss-cpu` flows work.

## Code of Conduct
By participating, you agree to follow our **Code of Conduct** (see `CODE_OF_CONDUCT.md`).

Happy hacking!
