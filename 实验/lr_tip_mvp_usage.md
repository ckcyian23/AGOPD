# LR-TIP MVP offline validation

本实验只做 offline token importance evaluation，不训练模型。

## 1. 准备环境

本项目在 `pyproject.toml` 中把 Windows/Linux 的 `torch` 指向 PyTorch CUDA 12.8 wheel index。`uv.lock` 不提交到仓库，避免在 CPU 机器上生成的 lock 文件导致 GPU 机器安装 `torch+cpu`。

```powershell
uv sync
```

同步后先检查 PyTorch 是否为 CUDA 版：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

期望看到类似：

```text
2.x.x+cu128
True
2
```

如果误装成 `+cpu`，直接重建环境：

```powershell
Remove-Item -Recurse -Force .venv
Remove-Item -Force uv.lock -ErrorAction SilentlyContinue
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果迁移机器没有使用 uv，也可以按 `pyproject.toml` 安装 `torch`、`transformers`、`accelerate`、`datasets`、`numpy`、`tqdm`。

## 2. Prompt 集

仓库提供了 128 条数学/推理 prompt：

```text
data/prompts.jsonl
```

默认不传 `--prompts-file` 时，只使用代码里的 4 条内置 prompt，适合检查链路是否能跑通。

## 3. 本地/低显存冒烟跑法

默认模型路径已经指向本仓库的本地模型。4GB 显存优先使用 `student-cache`：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 20 `
  --max-new-tokens 64 `
  --execution-mode student-cache `
  --output-chunk-size 4
```

`student-cache` 会先加载 student，生成 rollout 并缓存 student logits/hidden 到 CPU，然后释放 student，再加载 teacher。它更慢，但避免同时加载 Qwen3-0.6B 和 Qwen3-1.7B。

## 4. 24GB 单卡正式跑法

单张 3090 24GB 可以直接用 `joint`，同时加载 teacher/student，速度比 `student-cache` 更快：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --teacher-model models/Qwen/Qwen3-1.7B `
  --student-model models/Qwen/Qwen3-0.6B `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --window 16 `
  --budgets 0.05,0.1,0.2 `
  --output-dir outputs/lr_tip_mvp_128_joint
```

如果只想先冒烟：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 16 `
  --max-new-tokens 64 `
  --execution-mode joint `
  --device cuda `
  --dtype float16
```

## 5. 双 3090 并行跑法

两张 3090 每卡 24GB 时，推荐开两个进程，每个进程绑定一张卡并处理一半 prompt。PowerShell 示例：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --num-shards 2 `
  --shard-index 0 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2
```

另开一个 PowerShell 窗口：

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --num-shards 2 `
  --shard-index 1 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2
```

默认输出会自动分开：

```text
outputs/lr_tip_mvp/shard_0_of_2/report.json
outputs/lr_tip_mvp/shard_1_of_2/report.json
```

## 6. Prompt 文件格式

支持 `.txt`、`.jsonl`、`.json`。

`.txt` 每行一个 prompt。

`.jsonl` 每行可使用：

```json
{"prompt": "Solve 3x + 5 = 20. Show the reasoning."}
```

也兼容 `question` 或 `text` 字段。

## 7. 输出

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
