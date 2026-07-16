# LR-TIP MVP offline validation

本实验只做 offline token importance evaluation，不训练模型。

## 1. 准备环境

```powershell
uv sync
```

如果迁移机器没有使用 uv，也可以按 `pyproject.toml` 安装 `torch`、`transformers`、`accelerate`、`datasets`、`numpy`、`tqdm`。

## 2. 本地/低显存冒烟跑法

默认模型路径已经指向本仓库的本地模型：

```powershell
uv run lr-tip-eval --limit 4 --max-new-tokens 64 --execution-mode student-cache
```

`student-cache` 会先加载 student，生成 rollout 并缓存 student logits/hidden 到 CPU，然后释放 student，再加载 teacher。它更慢，但避免同时加载 Qwen3-0.6B 和 Qwen3-1.7B。

## 3. 大显存正式跑法

```powershell
uv run lr-tip-eval `
  --teacher-model models/Qwen/Qwen3-1.7B `
  --student-model models/Qwen/Qwen3-0.6B `
  --prompts-file data/prompts.jsonl `
  --limit 1000 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --window 16 `
  --budgets 0.1,0.2
```

`joint` 会同时加载 teacher/student，速度更快，适合迁移到大显存机器后使用。

## 4. Prompt 文件格式

支持 `.txt`、`.jsonl`、`.json`。

`.txt` 每行一个 prompt。

`.jsonl` 每行可使用：

```json
{"prompt": "Solve 3x + 5 = 20. Show the reasoning."}
```

也兼容 `question` 或 `text` 字段。

## 5. 输出

默认写入：

```text
outputs/lr_tip_mvp/report.json
```

核心指标：

- `mean_do_dr_corr`: D_O 和 D_R 的相关性，目标小于 0.7。
- `tip_lr_tip_overlap`: TIP 和 LR-TIP Top-k token 交集比例，越低说明 D_R 提供了新信息。
- `random_do_mean` / `tip_do_mean` / `lr_tip_do_mean`: 选中 token 的 teacher correction density。
- `random_dr_mean` / `tip_dr_mean` / `lr_tip_dr_mean`: 选中 token 的 relation discrepancy density。

MVP 成功信号：

- LR-TIP 选中 token 的 D_O 均值高于 random。
- LR-TIP 选中 token 的 D_R 均值高于 TIP 或 random。
- D_O/D_R correlation 不接近 1。
- LR-TIP 和 TIP overlap 不过高。
