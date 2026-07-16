# Entropy-aware LR-TIP Offline Results

更新时间：2026-07-17

## 1. 为什么重跑

之前的实验把 TIP baseline 简化成 KL-only，这是不完整的。

TIP 实际包含两个轴：

- student entropy
- teacher-student divergence

完整 TIP selection 使用 Soft-OR：

```text
TIP Soft-OR = 1 - (1 - H_norm) * (1 - D_norm)
```

因此本轮重跑 entropy-aware report，新增：

- entropy-only
- divergence-only
- TIP Soft-OR
- LR-TIP relation-only
- LR-TIP full Soft-OR

## 2. 实验配置

远程输出：

```text
/data_b/qtwei/ckcyi/AGOPD/outputs/lr_tip_mvp_128_entropy_fixed/report.json
```

配置：

```text
teacher_model: models/Qwen/Qwen3-1.7B
student_model: models/Qwen/Qwen3-0.6B
prompts_file: data/prompts.jsonl
limit: 128
max_new_tokens: 128
window: 16
layer_index: -2
device: cuda
dtype: float16
execution_mode: joint
output_kl_direction: forward
budgets: 0.05, 0.1, 0.2
```

## 3. 相关性

| Metric | Mean correlation |
| --- | ---: |
| corr(D_O, D_R) | 0.0481 |
| corr(entropy, D_O) | 0.5999 |
| corr(entropy, D_R) | 0.0419 |

解释：

- `D_R` 与 `D_O` 低相关。
- `D_R` 与 entropy 也低相关。
- entropy 与 `D_O` 中等相关，说明 TIP 的两个轴并非完全独立，但也不是同一个信号。

## 4. Aggregate 指标

### Budget 0.05

| Selector | D_O | D_R | Entropy |
| --- | ---: | ---: | ---: |
| Random | 0.1851 | 0.000653 | 0.4051 |
| Entropy-only | 0.8539 | 0.000723 | 2.8255 |
| Divergence-only | 1.7039 | 0.000784 | 1.6755 |
| TIP Soft-OR | 1.0807 | 0.000762 | 2.7551 |
| LR-TIP relation-only | 1.3420 | 0.002483 | 1.3081 |
| LR-TIP full Soft-OR | 1.0263 | 0.001594 | 2.5780 |

### Budget 0.10

| Selector | D_O | D_R | Entropy |
| --- | ---: | ---: | ---: |
| Random | 0.1903 | 0.000626 | 0.5028 |
| Entropy-only | 0.7445 | 0.000702 | 2.3604 |
| Divergence-only | 1.2255 | 0.000769 | 1.7089 |
| TIP Soft-OR | 0.9499 | 0.000705 | 2.3015 |
| LR-TIP relation-only | 1.0252 | 0.001822 | 1.3456 |
| LR-TIP full Soft-OR | 0.9167 | 0.001239 | 2.2042 |

### Budget 0.20

| Selector | D_O | D_R | Entropy |
| --- | ---: | ---: | ---: |
| Random | 0.2402 | 0.000638 | 0.6861 |
| Entropy-only | 0.6065 | 0.000662 | 1.8626 |
| Divergence-only | 0.8333 | 0.000710 | 1.5807 |
| TIP Soft-OR | 0.7379 | 0.000672 | 1.8313 |
| LR-TIP relation-only | 0.7212 | 0.001422 | 1.2457 |
| LR-TIP full Soft-OR | 0.7179 | 0.001038 | 1.7744 |

## 5. LR-TIP Full vs TIP Soft-OR

| Budget | LR-TIP full D_R > TIP count | LR-TIP full D_O > random count | TIP/LR-TIP full overlap | Mean D_R ratio |
| --- | ---: | ---: | ---: | ---: |
| 0.05 | 75 / 128 | 128 / 128 | 0.8737 | 2.28x |
| 0.10 | 94 / 128 | 128 / 128 | 0.8978 | 1.83x |
| 0.20 | 114 / 128 | 128 / 128 | 0.9026 | 1.54x |

解释：

- LR-TIP full Soft-OR 在所有 budget 下都保留了高于 random 的 `D_O`。
- LR-TIP full Soft-OR 的平均 `D_R` 明显高于 TIP Soft-OR。
- 但是 overlap 很高，说明 entropy 轴在 Soft-OR 中主导了大部分 selection。

## 6. 关键判断

当前结果支持：

1. `D_R` 是独立互补信号：它与 entropy 和 divergence 都低相关。
2. 加入 `D_R` 后，full Soft-OR 的 relation discrepancy density 明显提高。
3. full Soft-OR 与 TIP overlap 高，说明它是温和增强，不是大幅改变 token set。
4. relation-only LR-TIP 选择的 `D_R` 最强，但 entropy coverage 明显低于 TIP。

当前结果提示：

> 最终方法可能需要在 TIP Soft-OR 与 relation-only LR-TIP 之间做更细的融合，而不是简单三轴 Soft-OR 就结束。

可继续验证：

- window size 是否影响 `D_R` 提升。
- layer choice 是否影响 `D_R` 提升。
- reverse KL vs forward KL 是否影响 TIP/LR-TIP ranking。
- 是否需要引入可调融合：

```text
S = SoftOR(H_norm, D_norm, gamma * R_norm)
```

或：

```text
S = TIP_SoftOR + lambda * R_norm
```

## 7. 与旧结果的关系

旧 128 prompt 结果仍可说明：

- divergence-only 与 relation-only 的互补性强。
- `D_R` 能显著改变 token selection。

但旧结果不能再写作“完整 TIP baseline 对比”。

新的主要 baseline 应以本文件的 `TIP Soft-OR` 为准。

## 8. Batch A 消融结果

Batch A 比较：

- forward KL, window 16, layer -2
- reverse KL, window 16, layer -2
- forward KL, window 8, layer -2
- forward KL, window 32, layer -2

### 8.1 相关性

| Run | corr(D_O,D_R) | corr(entropy,D_O) | corr(entropy,D_R) |
| --- | ---: | ---: | ---: |
| forward / w16 | 0.0481 | 0.5999 | 0.0419 |
| reverse / w16 | 0.0381 | 0.6082 | 0.0419 |
| forward / w8 | 0.0401 | 0.5999 | 0.0294 |
| forward / w32 | 0.0563 | 0.5999 | 0.0691 |

结论：

- `D_R` 与 output divergence 在不同 KL direction / window 下都保持低相关。
- `D_R` 与 entropy 也保持低相关。
- 这支持 relation signal 的互补性。

### 8.2 LR-TIP Full 相对 TIP Soft-OR 的 D_R 提升

| Run | Budget 0.05 | Budget 0.10 | Budget 0.20 |
| --- | ---: | ---: | ---: |
| forward / w16 | 2.09x | 1.76x | 1.54x |
| reverse / w16 | 2.02x | 1.65x | 1.51x |
| forward / w8 | 2.16x | 1.85x | 1.60x |
| forward / w32 | 1.92x | 1.67x | 1.49x |

结论：

- window 8 的 D_R 提升略强。
- window 32 的 D_R 提升略弱，但仍明显高于 TIP。
- KL direction 不改变主要结论。

### 8.3 TIP / LR-TIP Full overlap

| Run | Budget 0.05 | Budget 0.10 | Budget 0.20 |
| --- | ---: | ---: | ---: |
| forward / w16 | 0.874 | 0.898 | 0.903 |
| reverse / w16 | 0.893 | 0.908 | 0.907 |
| forward / w8 | 0.876 | 0.895 | 0.898 |
| forward / w32 | 0.888 | 0.901 | 0.905 |

结论：

- LR-TIP Full 是 TIP Soft-OR 的温和增强，token set 大部分重合。
- relation-only LR-TIP 改变 selection 更强，但牺牲 entropy coverage。
- 如果论文需要更明显地区分 LR-TIP 与 TIP，后续可以探索：

```text
S = TIP_SoftOR + lambda * R_norm
```

或：

```text
S = 1 - (1 - H_norm) * (1 - D_norm) * (1 - gamma * R_norm)
```

当前代码尚未加入 `gamma/lambda`，这是下一步方法设计候选。

## 9. Batch B 消融结果

Batch B 已完成：

- layer -1
- layer -4
- layer -8
- alpha=0.3, beta=0.7
- alpha=0.7, beta=0.3

### 9.1 Layer 消融

| Run | corr(D_O,D_R) | corr(entropy,D_R) | D_R ratio @0.05 | D_R ratio @0.10 | D_R ratio @0.20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer -1 | 0.1353 | 0.2791 | 1.30x | 1.32x | 1.22x |
| layer -2 | 0.0481 | 0.0419 | 2.09x | 1.76x | 1.54x |
| layer -4 | 0.1107 | 0.1184 | 1.65x | 1.48x | 1.38x |
| layer -8 | 0.1046 | 0.1220 | 1.51x | 1.44x | 1.38x |

解释：

- `layer=-2` 当前最干净：与 entropy / divergence 相关性最低，且 LR-TIP full 相对 TIP 的 `D_R` 提升更强。
- `layer=-1` 的 raw `D_R` 数值更大，但与 entropy 的相关性升高，互补性变弱。
- `layer=-4` / `layer=-8` 介于二者之间，但都不如 `layer=-2`。

当前倾向：

> 主实验继续使用 `layer=-2`。

### 9.2 alpha/beta 消融

注意：

- alpha/beta 只影响旧的 relation-only zscore score：

```text
importance = alpha * z(D_O) + beta * z(D_R)
```

- alpha/beta 不影响 `LR-TIP full Soft-OR`，因为 full score 当前是 parameter-free 三轴 Soft-OR。

| Run | Budget | relation-only D_R | relation-only overlap with TIP |
| --- | ---: | ---: | ---: |
| alpha=0.3 beta=0.7 | 0.05 | 0.003036 | 0.184 |
| alpha=0.3 beta=0.7 | 0.10 | 0.002150 | 0.318 |
| alpha=0.3 beta=0.7 | 0.20 | 0.001579 | 0.445 |
| alpha=0.7 beta=0.3 | 0.05 | 0.001740 | 0.337 |
| alpha=0.7 beta=0.3 | 0.10 | 0.001435 | 0.533 |
| alpha=0.7 beta=0.3 | 0.20 | 0.001231 | 0.656 |

解释：

- beta 更大时，relation-only LR-TIP 选中的 token 与 TIP 更不同，`D_R` density 更高。
- 这说明 relation signal 本身很强，但如果直接三轴 Soft-OR，会被 entropy 轴冲淡。

下一步方法候选：

```text
LR-TIP add:
S = TIP_SoftOR + lambda * R_norm
```

或：

```text
LR-TIP gated:
S = TIP_SoftOR + lambda * R_norm * (1 - TIP_SoftOR)
```

这样可以在保持 TIP coverage 的同时，更显式地提升 relation-only 补充 token。

## 10. 长输出验证

已启动 128 prompts、`max_new_tokens=256` 的更大输出实验：

| GPU | Session | Output |
| --- | --- | --- |
| 1 | `lr_tip_256_main` | `outputs/lr_tip_mvp_128_256tok_main` |
| 2 | `lr_tip_256_window8` | `outputs/lr_tip_mvp_128_256tok_window8` |
| 3 | `lr_tip_256_reverse` | `outputs/lr_tip_mvp_128_256tok_reverse` |
| 0 | `lr_tip_256_layer_m4` | `outputs/lr_tip_mvp_128_256tok_layer_m4` |

目的：

- 检查结论在更长 rollout 上是否稳定。
- 观察 entropy/TIP dominance 是否随长度变化。
- 观察 window=8 在更长输出中是否仍优于 window=16。

### 10.1 已完成的 256-token 结果

已完成：

- `lr_tip_256_main`
- `lr_tip_256_window8`
- `lr_tip_256_reverse`

| Run | corr(D_O,D_R) | corr(entropy,D_O) | corr(entropy,D_R) |
| --- | ---: | ---: | ---: |
| 256 / forward / w16 | 0.0437 | 0.5882 | 0.0324 |
| 256 / forward / w8 | 0.0389 | 0.5882 | 0.0269 |
| 256 / reverse / w16 | 0.0307 | 0.6041 | 0.0324 |

| Run | D_R ratio @0.05 | D_R ratio @0.10 | D_R ratio @0.20 |
| --- | ---: | ---: | ---: |
| 256 / forward / w16 | 2.18x | 1.80x | 1.62x |
| 256 / forward / w8 | 2.28x | 1.88x | 1.68x |
| 256 / reverse / w16 | 2.07x | 1.76x | 1.61x |

| Run | overlap @0.05 | overlap @0.10 | overlap @0.20 |
| --- | ---: | ---: | ---: |
| 256 / forward / w16 | 0.898 | 0.910 | 0.905 |
| 256 / forward / w8 | 0.894 | 0.900 | 0.897 |
| 256 / reverse / w16 | 0.907 | 0.916 | 0.911 |

结论：

- 256-token 长输出下，`D_R` 与 entropy / divergence 仍低相关。
- LR-TIP full 相对 TIP Soft-OR 的 `D_R` 提升仍稳定存在。
- window=8 在 128-token 和 256-token 两种设置下都略强。
- reverse KL 不改变核心结论。

仍在运行：

- `lr_tip_256_layer_m4`
- `lr_tip_256_layer_last`
- `lr_tip_256_layer_m8`
- `lr_tip_256_alpha03`

### 10.2 256-token layer / alpha 消融

已完成：

- `lr_tip_256_layer_m4`
- `lr_tip_256_layer_last`
- `lr_tip_256_layer_m8`
- `lr_tip_256_alpha03`
- `lr_tip_256_alpha07`

| Run | corr(D_O,D_R) | corr(entropy,D_R) | D_R ratio @0.05 | D_R ratio @0.10 | D_R ratio @0.20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 / layer -1 | 0.1097 | 0.2496 | 1.26x | 1.26x | 1.22x |
| 256 / layer -2 | 0.0437 | 0.0324 | 2.18x | 1.80x | 1.62x |
| 256 / layer -4 | 0.0734 | 0.0742 | 1.63x | 1.56x | 1.49x |
| 256 / layer -8 | 0.0695 | 0.0839 | 1.55x | 1.59x | 1.54x |

结论：

- 256-token 下仍然是 `layer=-2` 最强、最干净。
- `layer=-1` 与 entropy 相关性明显偏高，不适合作为主设置。
- 更早层可以提升 raw relation discrepancy，但互补性弱于 `layer=-2`。

relation-only alpha/beta：

| Run | Budget | relation-only D_R | relation-only overlap with TIP |
| --- | ---: | ---: | ---: |
| 256 / alpha=0.3 beta=0.7 | 0.05 | 0.003390 | 0.201 |
| 256 / alpha=0.3 beta=0.7 | 0.10 | 0.002453 | 0.313 |
| 256 / alpha=0.3 beta=0.7 | 0.20 | 0.001791 | 0.438 |
| 256 / alpha=0.7 beta=0.3 | 0.05 | 0.001956 | 0.402 |
| 256 / alpha=0.7 beta=0.3 | 0.10 | 0.001667 | 0.542 |
| 256 / alpha=0.7 beta=0.3 | 0.20 | 0.001422 | 0.670 |

结论：

- 更长输出下，beta-heavy relation-only selector 仍明显改变 token set。
- 这进一步说明 `D_R` 是可用的补充信号。

### 10.3 512-token 验证

已完成：

- `lr_tip_512_main`
- `lr_tip_512_window8`
- `lr_tip_512_reverse`
- `lr_tip_512_layer_m4`

| Run | corr(D_O,D_R) | corr(entropy,D_O) | corr(entropy,D_R) |
| --- | ---: | ---: | ---: |
| 512 / forward / w16 / layer -2 | 0.0662 | 0.5862 | 0.0658 |
| 512 / forward / w8 / layer -2 | 0.0609 | 0.5862 | 0.0593 |
| 512 / reverse / w16 / layer -2 | 0.0643 | 0.6145 | 0.0658 |
| 512 / forward / w16 / layer -4 | 0.0646 | 0.5862 | 0.0516 |

| Run | D_R ratio @0.05 | D_R ratio @0.10 | D_R ratio @0.20 |
| --- | ---: | ---: | ---: |
| 512 / forward / w16 / layer -2 | 1.87x | 1.66x | 1.54x |
| 512 / forward / w8 / layer -2 | 1.95x | 1.70x | 1.60x |
| 512 / reverse / w16 / layer -2 | 1.78x | 1.63x | 1.53x |
| 512 / forward / w16 / layer -4 | 1.66x | 1.57x | 1.53x |

| Run | overlap @0.05 | overlap @0.10 | overlap @0.20 |
| --- | ---: | ---: | ---: |
| 512 / forward / w16 / layer -2 | 0.928 | 0.925 | 0.903 |
| 512 / forward / w8 / layer -2 | 0.922 | 0.923 | 0.896 |
| 512 / reverse / w16 / layer -2 | 0.933 | 0.929 | 0.906 |
| 512 / forward / w16 / layer -4 | 0.907 | 0.906 | 0.883 |

结论：

- 512-token 长输出下，`D_R` 与 entropy / divergence 仍保持低相关。
- LR-TIP full 相对 TIP Soft-OR 的 `D_R` 提升仍稳定存在，约 1.5x 到 2.0x。
- window=8 仍略强于 window=16。
- reverse KL 不改变主要结论。
- layer -4 的 raw `D_R` 更大，但 overlap 更低；主设置仍倾向 `layer=-2`。

## 11. 1024-prompt 规模验证

已生成并同步：

```text
data/prompts_1024.jsonl
```

数据规模：

```text
1024 deterministic synthetic math/reasoning prompts
```

启动脚本：

```text
scripts/launch_lr_tip_1024_256_shards.sh
```

当前运行：

| GPU | Sessions | Output |
| --- | --- | --- |
| 0 | `lr_tip_1024_s0`, `lr_tip_1024_s1` | `outputs/lr_tip_1024_256_shards/shard_0`, `shard_1` |
| 1 | `lr_tip_1024_s2`, `lr_tip_1024_s3` | `outputs/lr_tip_1024_256_shards/shard_2`, `shard_3` |
| 2 | `lr_tip_1024_s4`, `lr_tip_1024_s5` | `outputs/lr_tip_1024_256_shards/shard_4`, `shard_5` |
| 3 | `lr_tip_1024_s6`, `lr_tip_1024_s7` | `outputs/lr_tip_1024_256_shards/shard_6`, `shard_7` |

配置：

```text
prompts_file: data/prompts_1024.jsonl
limit: 1024
num_shards: 8
max_new_tokens: 256
window: 16
layer_index: -2
execution_mode: joint
device: cuda
dtype: float16
output_kl_direction: forward
budgets: 0.05,0.1,0.2
```

启动后健康检查：

```text
GPU0-3: about 9937 MiB each, 99% utilization
```

完成后合并：

```bash
cd /data_b/qtwei/ckcyi/AGOPD
.venv/bin/python -m agopd.experiments.merge_lr_tip_reports \
  --output outputs/lr_tip_1024_256_shards/merged/report.json \
  outputs/lr_tip_1024_256_shards/shard_0/report.json \
  outputs/lr_tip_1024_256_shards/shard_1/report.json \
  outputs/lr_tip_1024_256_shards/shard_2/report.json \
  outputs/lr_tip_1024_256_shards/shard_3/report.json \
  outputs/lr_tip_1024_256_shards/shard_4/report.json \
  outputs/lr_tip_1024_256_shards/shard_5/report.json \
  outputs/lr_tip_1024_256_shards/shard_6/report.json \
  outputs/lr_tip_1024_256_shards/shard_7/report.json
```

待验证：

- 1024 prompts 下 `D_R` 是否仍与 entropy / `D_O` 低相关。
- LR-TIP full 是否仍稳定提高 `D_R` density。
- full Soft-OR 与 TIP overlap 是否继续接近但小于 1。
- 规模变大后 window=8 是否仍值得作为主设置候选。

## 12. 下一步融合策略验证

当前代码已加入三种 full LR-TIP ranking：

```text
soft_or: S = SoftOR(H_norm, D_norm, gamma * R_norm)
add:     S = TIP_SoftOR + lambda * R_norm
gated:   S = TIP_SoftOR + lambda * R_norm * (1 - TIP_SoftOR)
```

默认仍是 `soft_or`，因此已完成和正在运行的结果不受影响。

待跑矩阵：

| Run | Purpose |
| --- | --- |
| `--lr-tip-full-mode soft_or --relation-gamma 0.5` | 检查降低 relation 强度后是否仍有稳定收益 |
| `--lr-tip-full-mode soft_or --relation-gamma 2.0` | 检查加强 relation 后 overlap 是否下降 |
| `--lr-tip-full-mode add --relation-lambda 0.25` | 轻量 additive 补充 relation signal |
| `--lr-tip-full-mode add --relation-lambda 0.5` | 中等 additive 补充 |
| `--lr-tip-full-mode gated --relation-lambda 0.5` | 只在 TIP 未覆盖充分的位置补 relation |
| `--lr-tip-full-mode gated --relation-lambda 1.0` | gated 强补充，对比 parameter-free Soft-OR |

优先级：

1. 等 1024-shard main 合并完成。
2. 如果 1024 结论稳定，先在 128 prompts / 256 tokens 上跑融合小矩阵。
3. 从小矩阵选 1-2 个最好的融合设置，再扩到 1024 prompts。
