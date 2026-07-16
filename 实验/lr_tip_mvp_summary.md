# LR-TIP 当前成果总结与下一步计划

更新时间：2026-07-16

## 1. 当前目标

本阶段目标不是训练模型，而是完成 LR-TIP 的 offline token importance validation：

> 在不训练的情况下，验证 LR-TIP 选出的 token 是否比 random 更有蒸馏价值，并且是否比 TIP 额外捕捉到 teacher/student hidden relation 的差异。

重要修正：

> 之前把 TIP 理解成 KL-only 是不完整的。TIP 实际上是 student entropy 和 teacher-student divergence 的二维 taxonomy，完整 selection score 使用 entropy + divergence 的 Soft-OR。当前旧结果只能视为 divergence-only baseline 与 LR-TIP relation signal 的对比，不能再写成完整 TIP baseline 对比。

当前 LR-TIP 采用三个可分析信号：

- `D_H`: student entropy，即学生在当前位置的不确定性。
- `D_O`: output disagreement，即 teacher/student logits 分布的 token-level KL 或 divergence ranking signal。
- `D_R`: relation disagreement，即 teacher/student hidden states 的局部 token 关系差异。

旧版 divergence-relation 融合方式：

```text
importance = 0.5 * zscore(D_O) + 0.5 * zscore(D_R)
```

新增 entropy-aware full score：

```text
TIP Soft-OR    = 1 - (1 - H_norm) * (1 - D_norm)
LR-TIP Soft-OR = 1 - (1 - H_norm) * (1 - D_norm) * (1 - R_norm)
```

## 2. 已完成内容

代码部分：

- 实现 LR-TIP scorer：`agopd/lr_tip.py`
- 实现 offline evaluation 指标：`agopd/offline_eval.py`
- 实现实验入口：`agopd/experiments/offline_lr_tip_eval.py`
- 支持 prompt 文件输入：`data/prompts.jsonl`
- 支持多 budget 对比：`0.05, 0.1, 0.2`
- 支持 prompt sharding：`--num-shards` / `--shard-index`
- 添加 pytest 单测：`tests/test_lr_tip.py`
- 修复 uv 环境配置，使 Windows/Linux 默认使用 PyTorch CUDA wheel index。

数据部分：

- 已生成 128 条数学/推理 prompts：`data/prompts.jsonl`
- 已完成 4 prompt 冒烟测试。
- 已完成 20 prompt 本地测试。
- 已完成 128 prompt offline validation。
- 2026-07-17 修正：旧 validation 缺少 entropy 轴，需重新跑 entropy-aware report。

## 3. 最新完整实验

结果文件：

```text
outputs/lr_tip_mvp_128_student/report.json
```

备注：该结果文件是从原输出目录迁移过来的，report 内部 `config.output_dir` 仍显示为旧目录名，但当前有效保存路径是上面的 `_student` 目录。

实验配置：

```text
teacher_model: models/Qwen/Qwen3-1.7B
student_model: models/Qwen/Qwen3-0.6B
prompts_file: data/prompts.jsonl
num_prompts: 128
max_new_tokens: 128
window: 16
layer_index: -2
alpha: 0.5
beta: 0.5
budgets: 0.05, 0.1, 0.2
device: cuda
dtype: float16
execution_mode: student-cache
```

说明：本次完整结果使用的是 `student-cache`，不是 `joint`。这不影响指标含义，只影响运行速度。

## 4. 关键结果

整体相关性：

```text
mean corr(D_O, D_R) = 0.0481
min corr = -0.2390
max corr = 0.6516
```

解释：

- 平均相关性非常低，说明 `D_R` 不是 `D_O` 的重复信息。
- 最高单样本相关性仍小于 0.7，整体上可以认为二者具有互补性。

Aggregate 指标：

| Budget | Selected tokens | Random D_O | TIP D_O | LR-TIP D_O | Random D_R | TIP D_R | LR-TIP D_R | TIP/LR-TIP overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 6 | 0.1880 | 1.7284 | 1.3577 | 0.000685 | 0.000796 | 0.002593 | 0.6016 |
| 0.10 | 13 | 0.1890 | 1.2461 | 1.0455 | 0.000632 | 0.000795 | 0.001877 | 0.6581 |
| 0.20 | 26 | 0.2376 | 0.8450 | 0.7340 | 0.000678 | 0.000741 | 0.001468 | 0.6728 |

逐样本胜率：

| Budget | LR-TIP D_O > Random | LR-TIP D_R > TIP | LR-TIP D_R > Random |
| --- | ---: | ---: | ---: |
| 0.05 | 127 / 128 | 124 / 128 | 128 / 128 |
| 0.10 | 128 / 128 | 128 / 128 | 128 / 128 |
| 0.20 | 128 / 128 | 128 / 128 | 128 / 128 |

`D_R` 相对 TIP 的平均提升倍数：

| Budget | Mean LR-TIP D_R / TIP D_R |
| --- | ---: |
| 0.05 | 3.60x |
| 0.10 | 2.59x |
| 0.20 | 2.06x |

## 5. 当前结论

旧版 offline validation 仍然支持以下较弱结论：

1. `D_R` 与 `D_O` 低相关，LR-TIP 的 relation signal 不是 TIP KL signal 的简单重复。
2. LR-TIP 选中 token 的 `D_O` 明显高于 random，说明它仍保留 teacher correction value。
3. LR-TIP 选中 token 的 `D_R` 明显高于 divergence-only 和 random，说明局部关系项确实改变了 token selection 行为。
4. LR-TIP 与 divergence-only baseline 的 overlap 在 `0.60-0.67` 左右，说明二者有交集但不是同一套 token。

一句话总结：

> LR-TIP 的 relation signal MVP 已成立；但完整 TIP 对比需要补 entropy-aware baseline 后重跑。

## 6. 当前局限

目前还没有验证：

- LR-TIP full Soft-OR 是否优于 TIP Soft-OR。
- 使用 LR-TIP mask 训练后的 student 是否在 GSM8K / MATH-500 / general reasoning 上涨分。
- `alpha=0.5, beta=0.5` 是否最优。
- `window=16` 是否最优。
- `layer_index=-2` 是否最优。
- `D_R` 是否在更大规模、更长输出、更复杂数据上保持稳定优势。

另外，本阶段只验证 token selection quality，不等价于最终训练收益。

## 7. 下一步计划

### Step 0: 先重跑 entropy-aware offline evaluation

目标：修正 TIP baseline，对比：

- Random
- Entropy-only
- Divergence-only
- TIP Soft-OR: entropy + divergence
- LR-TIP relation-only: divergence + relation
- LR-TIP full Soft-OR: entropy + divergence + relation

推荐命令：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2 `
  --output-dir outputs/lr_tip_mvp_128_entropy_fixed
```

重点看新字段：

- `mean_entropy_do_corr`
- `mean_entropy_dr_corr`
- `entropy_only_*`
- `div_only_*`
- `tip_*`
- `lr_tip_full_*`
- `tip_lr_tip_full_overlap`

### Step 1: 做参数消融的 offline evaluation

目标：确认 LR-TIP 分数设计是否稳健。

建议先跑：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2 `
  --alpha 0.3 `
  --beta 0.7 `
  --output-dir outputs/lr_tip_mvp_128_alpha03_beta07
```

再跑：

```powershell
--alpha 0.7 --beta 0.3
--window 8
--window 32
--layer-index -1
--layer-index -2
--layer-index -4
```

重点看：

- `mean_do_dr_corr`
- `lr_tip_do_mean / random_do_mean`
- `lr_tip_dr_mean / tip_dr_mean`
- `tip_lr_tip_overlap`

### Step 2: 接入 OPD/TIP 训练 pipeline

目标：把当前 offline scorer 接到 token-level distillation loss。

需要做：

1. 在原 OPD/TIP token selection 位置接入 `compute_lr_tip`。
2. 用 LR-TIP importance 生成 top-k token mask。
3. 用该 mask 控制 token-level KL loss。
4. 保持原有 random / TIP baseline 可切换。

建议保留这些开关：

```text
--selector random
--selector tip
--selector lr-tip
--lr-tip-alpha
--lr-tip-beta
--lr-tip-window
--lr-tip-layer-index
--token-budget
```

### Step 3: 小规模训练验证

目标：先证明训练链路能跑通，不急着追最终 SOTA。

建议设置：

```text
teacher: Qwen3-1.7B
student: Qwen3-0.6B
data: 1k-5k reasoning prompts
token_budget: 0.1
selectors: random / TIP / LR-TIP
max_new_tokens: 128
```

评估：

- GSM8K
- MATH-500
- 训练 loss 曲线
- selected token 的 `D_O` / `D_R` 分布

成功标准：

- LR-TIP 不比 TIP 差。
- LR-TIP 在至少一个 reasoning benchmark 上优于 TIP 或 random。
- LR-TIP 训练过程稳定，没有 loss 爆炸或 mask 退化。

### Step 4: 扩大实验规模

如果 Step 3 成立，再扩大：

- prompts 从 1k-5k 扩到 10k+
- max_new_tokens 从 128 扩到 256
- budget 从 `0.05/0.1/0.2` 做系统比较
- 加入更多 reasoning / math / code 数据

### Step 5: 论文叙述方向

当前可以支撑的论文动机：

> TIP 主要关注输出分布差异，而 LR-TIP 进一步关注 teacher/student 在局部 token 关系结构上的差异。Offline validation 显示，relation disagreement 与 output disagreement 低相关，并且 LR-TIP 能在保持 teacher correction density 的同时显著提高 selected token 的 relation discrepancy。

后续训练实验要证明：

> 这种 relation-aware token selection 能转化为实际蒸馏收益。

## 8. 当前推荐下一条命令

如果只做下一轮 offline 消融，推荐：

```powershell
uv run python -m agopd.experiments.offline_lr_tip_eval `
  --prompts-file data/prompts.jsonl `
  --limit 128 `
  --max-new-tokens 128 `
  --execution-mode joint `
  --device cuda `
  --dtype float16 `
  --budgets 0.05,0.1,0.2 `
  --alpha 0.3 `
  --beta 0.7 `
  --output-dir outputs/lr_tip_mvp_128_alpha03_beta07
```

如果开始推进真正论文实验，下一步应优先做 OPD/TIP 训练 pipeline 集成。
