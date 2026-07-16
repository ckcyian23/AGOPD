# LR-TIP MVP results

## 2026-07-16 local smoke run

Purpose:

- Verify the offline LR-TIP scoring pipeline end to end.
- Check whether relation disagreement `D_R` is complementary to output disagreement `D_O`.
- Confirm LR-TIP selected tokens retain teacher correction density while increasing relation discrepancy.

Command shape:

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --limit 4 `
  --max-new-tokens 64 `
  --execution-mode student-cache
```

Config summary:

```text
teacher_model: models/Qwen/Qwen3-1.7B
student_model: models/Qwen/Qwen3-0.6B
num_prompts: 4
target_tokens_per_prompt: 64
window: 16
layer_index: -2
alpha: 0.5
beta: 0.5
budgets: 0.1, 0.2
execution_mode: student-cache
output_chunk_size: 8
```

Aggregate output:

| Budget | Random D_O | TIP D_O | LR-TIP D_O | Random D_R | TIP D_R | LR-TIP D_R | TIP/LR-TIP overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.1475 | 1.0688 | 0.8195 | 0.000366 | 0.000572 | 0.001603 | 0.5833 |
| 0.2 | 0.2860 | 0.7618 | 0.6761 | 0.000592 | 0.000690 | 0.001150 | 0.7115 |

`mean_do_dr_corr = 0.0773`.

Interpretation:

- `D_O` and `D_R` are weakly correlated, so the relation signal is not just a duplicate of TIP's KL signal.
- LR-TIP selected tokens have much higher `D_O` than random tokens, so the fused score keeps teacher correction value.
- LR-TIP selected tokens have clearly higher `D_R` than TIP/random, so the local relation term changes token selection in the intended direction.

## Next full run

Use the 128-prompt file:

```text
data/prompts.jsonl
```

Recommended single-3090 command:

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2 `
  --output-dir outputs/lr_tip_mvp_128_joint
```

Recommended dual-3090 command pair is documented in `实验/lr_tip_mvp_usage.md`.
