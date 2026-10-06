# Reproducing the offline baseline

```powershell
uv sync --extra dev
uv run ragx index .\path\to\documents --output cache\index.json
uv run ragx ask "your question" --index-path cache\index.json
uv run ragx benchmark
```

For retrieval evaluation, prepare JSONL rows containing `question` and `gold_chunk_ids`, then run `ragx eval`. The included code records environment and Git revision with result files. The repository intentionally makes no unmeasured quality claim.
