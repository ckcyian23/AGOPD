# AGOPD LR-TIP MVP

This repository contains a lightweight offline validation pipeline for LR-TIP
token importance scoring.

The current MVP does not train a model. It:

1. Generates student rollouts with Qwen3-0.6B.
2. Runs teacher/student forward passes on the same trajectory.
3. Computes token-level output disagreement `D_O`.
4. Computes local relation disagreement `D_R`.
5. Compares random, TIP, and LR-TIP token selection.

Main docs:

- Usage: `实验/lr_tip_mvp_usage.md`
- Latest smoke-run summary: `实验/lr_tip_mvp_results.md`
- Prompt set: `data/prompts.jsonl`

Quick 24GB GPU run:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
uv run pytest -q
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2
```

`pyproject.toml` points `torch` to the PyTorch CUDA 12.8 wheel index on
Windows/Linux. `uv.lock` is intentionally not committed because the correct
PyTorch wheel depends on the experiment machine's CUDA/driver setup.
