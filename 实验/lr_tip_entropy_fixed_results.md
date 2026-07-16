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

路线修正：

> 1024 synthetic prompt 结构过于类似，只能作为工程吞吐和稳定性参考，不再作为主论文证据。

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

## 12. 真实数据与 gradient-improvement 验证

已生成：

```text
data/real_prompts_512.jsonl
```

组成：

| Dataset | Count | Split |
| --- | ---: | --- |
| `openai/gsm8k` | 256 | test |
| `HuggingFaceH4/MATH-500` | 256 | test |

关键新增指标：

```text
grad_improvement(t) = sum(p.grad.float().square().sum()
                          for p in student.parameters())
```

解释：

- 对 token `t` 的 teacher/student logits 计算蒸馏 KL loss。
- 对 student 参数执行一次 `backward()`，但不更新。
- 每个 token 计算后必须 `zero_grad()`。
- 一阶泰勒展开下，固定学习率的单步 loss 下降量与梯度范数平方成正比。
- 如果 `D_R` 是有用的蒸馏选点信号，它应该与 `grad_improvement` 有正相关。
- 只看 `D_R` density 不够，必须看它是否对应真实训练收益 proxy。

当前待跑：

| Run | Output | Purpose |
| --- | --- | --- |
| grad real32 w16 | `outputs/lr_tip_grad_improvement_real32_w16` | 真实数据 gradient-improvement pilot |

启动脚本：

```text
scripts/launch_lr_tip_grad_improvement_real32.sh
```

判断标准：

1. `corr_relation_improvement` 是否为正。
2. `lr_tip_full_grad_improvement_mean` 是否高于 `tip_grad_improvement_mean` 和 random。
3. window8 是否改善 `D_R -> grad_improvement`，而不只是提高 `D_R` density。
4. 如果上述不成立，暂缓蒸馏，回头改 relation signal 或 fusion。

### 12.1 real32 / w16 初步结果

输出：

```text
outputs/lr_tip_grad_improvement_real32_w16/merged/report.json
```

配置：

```text
real prompts: 32
sampled tokens: 512
max_new_tokens: 64
tokens_per_prompt: 16
window: 16
layer_index: -2
improvement: per-token KL backward full-parameter grad norm squared
```

相关性：

| Score | Corr with grad_improvement |
| --- | ---: |
| output disagreement `D_O` | 0.539 |
| relation disagreement `D_R` | 0.306 |
| student entropy | 0.114 |
| TIP Soft-OR | 0.264 |
| LR-TIP full Soft-OR | 0.265 |

Top-budget mean `grad_improvement`：

| Budget | Random | Entropy | Divergence | Relation | TIP | LR-TIP full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 25,032 | 44,573 | 664,515 | 368,583 | 278,398 | 282,444 |
| 0.10 | 30,050 | 55,544 | 398,395 | 211,693 | 232,004 | 246,331 |
| 0.20 | 33,593 | 87,659 | 237,757 | 117,494 | 202,720 | 204,883 |

解释：

- `D_R` 对 gradient-improvement 有正相关，初步支持 relation signal 的训练收益意义。
- `D_O` 当前更强，说明 teacher/student output divergence 本身仍是最强单轴信号。
- `D_R` 单轴 selector 在低 budget 下选到的 gradient-improvement 明显高于 random/entropy，但低于 divergence。
- 当前 parameter-free 三轴 Soft-OR 只比 TIP 小幅更高，说明后续重点应验证 additive/gated fusion，而不是直接扩大数据或进入蒸馏。

正在运行：

```text
outputs/lr_tip_grad_improvement_real32_w16_fusion
```

目的：

- 同时记录 `lr_tip_soft_or` / `lr_tip_add` / `lr_tip_gated`。
- 判断更显式的 relation fusion 是否比 TIP Soft-OR 有更强训练收益选择能力。

### 12.2 real32 / w16 fusion 结果

输出：

```text
outputs/lr_tip_grad_improvement_real32_w16_fusion/merged/report.json
```

相关性：

| Score | Corr with grad_improvement |
| --- | ---: |
| `D_O` | 0.539 |
| `D_R` | 0.306 |
| entropy | 0.114 |
| TIP Soft-OR | 0.264 |
| LR-TIP Soft-OR | 0.265 |
| LR-TIP additive | 0.264 |
| LR-TIP gated | 0.265 |

Top-budget mean `grad_improvement`：

| Budget | Random | TIP | LR-TIP Soft-OR | LR-TIP additive | LR-TIP gated |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 25,033 | 278,396 | 282,441 | 396,545 | 282,441 |
| 0.10 | 30,050 | 232,004 | 246,330 | 245,790 | 246,330 |
| 0.20 | 33,592 | 202,720 | 204,883 | 187,234 | 204,883 |

解释：

- additive fusion 在 5% 低预算下明显优于 TIP 和 parameter-free Soft-OR。
- additive 在 20% 预算下变差，说明 relation signal 更像低预算补充信号，不适合无约束扩大覆盖。
- gated 与 Soft-OR 当前几乎相同，说明默认参数下还没有真正改变排序。
- 下一步需要比较 window8，并考虑更小 `lambda` 或只在低 budget 场景使用 additive LR-TIP。

正在运行：

```text
outputs/lr_tip_grad_improvement_real32_w8_fusion
```

### 12.3 real32 fusion sweep 结论

sweep 不需要重新 backward，直接读取 gradient report 的 per-token `TIP_SoftOR` 和 `relation_norm`，比较不同融合参数：

```text
soft_or: SoftOR(TIP, gamma * R_norm)
add:     TIP + lambda * R_norm
gated:   TIP + lambda * R_norm * (1 - TIP)
```

window 16 最优：

| Budget | Best fusion | Mean grad_improvement |
| --- | --- | ---: |
| 0.05 | `add_2.0` | 406,376 |
| 0.10 | `add_0.75` | 249,425 |
| 0.20 | `add_0.25` | 206,741 |

window 8 最优：

| Budget | Best fusion | Mean grad_improvement |
| --- | --- | ---: |
| 0.05 | `add_1.5` | 416,035 |
| 0.10 | `soft_or_1.5` / `gated_1.5` | 248,450 |
| 0.20 | `add_0.25` | 208,052 |

关键判断：

- relation signal 的价值主要出现在低预算 token selection。
- additive fusion 比 parameter-free Soft-OR 更能把 relation signal 推进 top selection。
- 大预算下需要更小的 lambda，否则 relation 会带来噪声。
- window8 在 5% / 20% budget 上略优，但 `corr_relation_improvement` 低于 window16；不能简单说 window8 更好。

路线修正：

> 下一步不继续扩大 proxy 数据；先做小规模真实更新验证，比较 random / TIP / LR-TIP additive 选点后，student 是否真的在 held-out loss 上有提升。

候选蒸馏设置：

| Selector | Reason |
| --- | --- |
| random | 下界 |
| TIP Soft-OR | 现有 baseline |
| LR-TIP add, `lambda=1.5`, window8, budget 0.05 | 低预算 proxy 最强 |
| LR-TIP add, `lambda=0.75`, window16, budget 0.10 | 中预算较稳 |
| LR-TIP add, `lambda=0.25`, window8/window16, budget 0.20 | 大预算较稳 |

## 14. 小规模蒸馏 uplift 验证

目的：

- gradient-improvement 是 proxy，不等于真实训练提升。
- 必须验证选中的 token 用于一次小规模 KL 蒸馏更新后，held-out fixed rollouts 上的 teacher KL 是否下降。

脚本：

```text
agopd/experiments/distillation_uplift_eval.py
```

当前已启动的 pilot：

```text
outputs/lr_tip_distill_uplift_real16_budget005
```

配置：

```text
train prompts: 16
eval prompts: 16
max_new_tokens: 64
budget: 0.05
epochs: 1
optimizer: SGD
lr: 1e-7
eval metric: held-out teacher KL on fixed base-student rollouts
```

对照：

| Selector | Output |
| --- | --- |
| random | `outputs/lr_tip_distill_uplift_real16_budget005/random` |
| divergence | `outputs/lr_tip_distill_uplift_real16_budget005/divergence` |
| TIP Soft-OR | `outputs/lr_tip_distill_uplift_real16_budget005/tip` |
| LR-TIP additive | `outputs/lr_tip_distill_uplift_real16_budget005/lr_tip_add_w8_lam15` |

判断标准：

- `eval_kl_delta < 0` 表示 held-out KL 改善。
- LR-TIP additive 需要至少优于 TIP 和 random，才值得进入更大训练。
- 如果所有 selector 都不改善，需要先调学习率/训练稳定性，而不是扩大数据。

### 14.1 split offset 0 / budget 0.05 结果

输出：

```text
outputs/lr_tip_distill_uplift_real16_budget005
```

| Selector | Eval KL before | Eval KL after | Delta | Relative delta | Selected tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.303044 | 0.302859 | -0.000185 | -0.061% | 48 / 1024 |
| divergence | 0.303044 | 0.302829 | -0.000216 | -0.071% | 48 / 1024 |
| TIP Soft-OR | 0.303044 | 0.302871 | -0.000173 | -0.057% | 48 / 1024 |
| LR-TIP additive, w8, lambda=1.5 | 0.303044 | 0.302746 | -0.000299 | -0.099% | 48 / 1024 |

解释：

- 四个 selector 都有轻微 held-out KL 改善，说明训练脚本和学习率没有明显跑坏。
- LR-TIP additive 的改善最大，强于 TIP、random 和 divergence。
- 绝对改善仍很小，不能直接写最终结论。
- 这一步支持继续做 split 复现，而不是立刻扩大模型/数据。

正在运行复现：

```text
outputs/lr_tip_distill_uplift_real16_budget005_offset32
```

配置与上表相同，但 `prompt_offset=32`。

### 14.2 split offset 32 / budget 0.05 结果

输出：

```text
outputs/lr_tip_distill_uplift_real16_budget005_offset32
```

| Selector | Eval KL before | Eval KL after | Delta | Relative delta | Selected tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.326766 | 0.326860 | +0.000094 | +0.029% | 48 / 1024 |
| divergence | 0.326766 | 0.326712 | -0.000054 | -0.017% | 48 / 1024 |
| TIP Soft-OR | 0.326766 | 0.326573 | -0.000193 | -0.059% | 48 / 1024 |
| LR-TIP additive, w8, lambda=1.5 | 0.326766 | 0.326774 | +0.000008 | +0.002% | 48 / 1024 |

解释：

- offset32 上，LR-TIP additive 没有复现 offset0 的 uplift，反而略微变差。
- TIP Soft-OR 在这个 split 上最好。
- 这说明 gradient-improvement proxy 虽然证明 `D_R` 有训练收益信号，但 5% budget 的 aggressive additive selection 不稳定。
- 当前不能声称 LR-TIP additive 已经稳定优于 TIP。

路线修正：

> 不继续扩大 5% additive 结果。改为验证更稳的中预算设置：budget 0.10、window16、lambda 0.75，并继续和 random / divergence / TIP 对照。

正在运行：

```text
outputs/lr_tip_distill_uplift_real16_budget010_offset32
```

### 14.3 split offset 32 / budget 0.10 结果

输出：

```text
outputs/lr_tip_distill_uplift_real16_budget010_offset32
```

| Selector | Eval KL before | Eval KL after | Delta | Relative delta | Selected tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.326766 | 0.326769 | +0.000004 | +0.001% | 96 / 1024 |
| divergence | 0.326766 | 0.326679 | -0.000087 | -0.027% | 96 / 1024 |
| TIP Soft-OR | 0.326766 | 0.326705 | -0.000061 | -0.019% | 96 / 1024 |
| LR-TIP additive, w16, lambda=0.75 | 0.326766 | 0.326731 | -0.000035 | -0.011% | 96 / 1024 |

解释：

- 中预算下 LR-TIP additive 仍然没有超过 TIP/divergence。
- divergence 在这个 split 上最稳。
- 这进一步说明 additive fusion 不能作为当前主方法。

路线修正：

> `D_R` 有 gradient-improvement 信号，但不能简单 additive 到 TIP ranking。下一步改为交集式 selector：只奖励同时具有高 TIP 与高 relation 的 token，避免 relation 把低 TIP token 强行顶上来。

待验证 selector：

```text
lr_tip_product = TIP_SoftOR * R_norm
lr_tip_product_add = TIP_SoftOR + lambda * TIP_SoftOR * R_norm
```

判断：

- 如果 product selector 比 TIP/divergence 更稳，说明 relation 应作为 gating/confirming signal。
- 如果 product 仍不稳，说明 `D_R` 当前更适合作为分析信号，不适合作为直接训练选点。

### 14.4 split offset 32 / product selector 结果

输出：

```text
outputs/lr_tip_distill_uplift_product_offset32
```

| Selector | Budget | Eval KL before | Eval KL after | Delta | Relative delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR-TIP product, w16 | 0.05 | 0.326766 | 0.326617 | -0.000149 | -0.046% |
| LR-TIP product, w16 | 0.10 | 0.326766 | 0.326911 | +0.000145 | +0.044% |

解释：

- product 5% 比 additive 5% 稳定，但仍弱于同 split 的 TIP 5%。
- product 10% 明显变差。
- relation 更适合低 budget 的确认信号，不适合扩大覆盖。

正在运行更保守 selector：

```text
outputs/lr_tip_distill_uplift_product_add_offset32
```

目的：

- 验证 `TIP + lambda * TIP * R_norm` 是否能保留 TIP 稳定性，同时带来 relation tie-break。
- 当前运行 lambda=0.25 和 lambda=0.5，budget=0.05。

### 14.5 split offset 32 / product-add 结果

输出：

```text
outputs/lr_tip_distill_uplift_product_add_offset32
```

| Selector | Budget | Lambda | Eval KL before | Eval KL after | Delta | Relative delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR-TIP product-add, w16 | 0.05 | 0.25 | 0.326766 | 0.326755 | -0.000011 | -0.003% |
| LR-TIP product-add, w16 | 0.05 | 0.50 | 0.326766 | 0.326685 | -0.000081 | -0.025% |

解释：

- product-add 比 aggressive additive 稳，但仍没有超过 TIP Soft-OR。
- 当前多个 split/预算说明：把 `D_R` 只作为 token ranking 公式的一部分，不能稳定带来 held-out KL uplift。

重大路线修正：

> 下一步不继续微调 ranking 公式。更合理的验证是 relation-aware training：在选中的 token 上训练 `output KL + mu * relation-profile loss`，并同时评估 held-out output KL 与 relation discrepancy 是否下降。

原因：

- `D_R` 衡量的是局部 hidden relation mismatch。
- 只用 output KL 更新，不一定会修复 hidden relation mismatch。
- 如果 LR-TIP 的核心是 relation signal，那么 relation 应该进入训练目标或辅助目标，而不是只进入 selection。

## 13. 下一步融合策略验证

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
