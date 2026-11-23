# 固定 token 路径与真实模型 KV 核验（2026-11-18）

本页保留固定路径实验当时的结论。后续尾块干预和固定上游复核结果见 [阶段 1 验收报告](STAGE1_ACCEPTANCE.md)。

**精确缓存/搬运检查全部通过；预先定义的冷/热数值筛查 6/144 个位置失败，阶段 1 仍未冻结。** 本轮从上次未完成的正确性项继续，没有重跑已通过的 baseline、prefill 或 scratch 搬运性能点。旧失败状态保持不变，没有通过扩大容差把本轮改为成功。

## 接手核对

仍在 `feat/pegaflow`、HEAD `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。本轮开始时，`results/20261117_validation.json` 中所有源码哈希都与工作区一致，没有遗留实验进程。复用原 `.venv`、本地模型及上轮成功工件；没有安装依赖、实现 CPU cache、提交或推送。本轮新增诊断/分析脚本与测试，推理引擎源码未再修改。

## 预先写定的协议

[CORRECTNESS_PROTOCOL.md](CORRECTNESS_PROTOCOL.md) 在 GPU 实验前保存，文档和脚本 SHA256 都进入 run 快照。它规定：

- 固定各请求的历史输出 token，再对齐同一个输出位置比较全部 151936 维 logits，排除自由生成分叉造成的输入差异。
- warm 复用完整块的 hash 链、token IDs、ref_count 和 KV bits 必须正确，且模型执行前后缓存前缀不变。
- 相同形状重复 warm、真实 KV round-trip 后的 warm，各位置 logits 均须逐位一致。
- 冷/热不同形状采用公开的工程筛查：`max_abs <= 2 * 2^-7 * scale`，`relative_L2 <= 4 * 2^-7 = 3.125%`。这是诊断范围，不是数学误差上界。

协议明确区分已知回归输入与新输入，也明确不以近并列分类替代精确检查。本轮运行后未改阈值。

## 输入与实测

模型/环境与上轮一致，完整元数据保存于 `results/20261118_fixed_path_probe/environment.json`。配置为 BF16、eager、KV cap=160、block=256、max batched tokens=2048、max model len=9216。诊断 hooks 包含同步和复制，**本轮任何耗时均不能作性能结论**。

| 组 | 输入来源 | 请求 × 输出位置 | 数值筛查失败 | cold/warm argmax 分歧 | 最大相对 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| regression_2048_c4 | 旧失败运行的四条输入与 cold 输出路径 | 4 × 16 | 3 / 64 | 2 / 64 | 5.179% |
| heldout_2048_c4 | 新 seed=20261118，先生成 cold 路径再冻结 | 4 × 16 | 2 / 64 | 0 / 64 | 4.222% |
| heldout_8192_c1 | 新 seed=20261119，先生成 cold 路径再冻结 | 1 × 16 | 1 / 16 | 0 / 16 | 5.195% |

每组实际执行 `cold → warm → warm_repeat → warm_after_roundtrip`，共 576 条输出位置记录，其中 144 对用于冷/热数值比较。所有组的 warm-repeat 和 round-trip logits 精确对照通过。

### 精确检查

- **864 次前缀 KV bits 检查通过**：每次 warm 执行前后，与 cold 计算所得完整可复用 prompt 块的 GPU 快照比较；同时检查 token、hash 链、引用状态、首次连续命中量和固定历史 token 路径。
- **87 个真实模型 KV 块 round-trip 通过**，不是随机 scratch KV。分别为旧 2048 组 28 块、新 2048 组 28 块、8192 组 31 块；每块 36 MiB，单方向累计 3132 MiB。
- 搬运时队列为空，所有目标块 ref_count=0；一次使用 36 MiB pinned scratch。先完成并验证 D2H，再将空闲 GPU 目标块临时填 NaN，确认其确实变化，最后 H2D 回原位并逐位验证。这样不能靠无效 H2D 得到假阳性。
- GPU cache 命中元数据未改，没有 CPU 保存策略、LRU、恢复发布或活跃块覆盖。诊断结束后 144 个位置的 warm logits 与 round-trip 前逐位一致。

这些检查支持所覆盖输入上的缓存映射、前缀保留和复制无损；不等同证明全部调度、所有输入或未来 CPU cache 实现正确。

### 数值筛查未通过的位置

request 和 position 均从 0 开始。

| 组 | request / position | 相对 L2 | max abs / 该位置 budget | argmax 一致 |
| --- | --- | ---: | ---: | --- |
| regression_2048_c4 | 2 / 5 | 3.234% | 0.269531 / 0.400391 | 是 |
| regression_2048_c4 | 2 / 11 | 5.179% | 0.265625 / 0.531250 | 是 |
| regression_2048_c4 | 2 / 14 | 3.416% | 0.312500 / 0.539063 | 是 |
| heldout_2048_c4 | 0 / 4 | 4.222% | 0.265625 / 0.414063 | 是 |
| heldout_2048_c4 | 0 / 5 | 4.154% | 0.218750 / 0.400391 | 是 |
| heldout_8192_c1 | 0 / 4 | 5.195% | 0.375000 / 0.357422 | 是 |

六处均超过相对 L2 范围；8192 位置还超过 max-abs 范围。两处 argmax 分歧反而通过数值筛查：旧请求 0 的位置 0、13，分别是原先并列的 44/198，以及相差 0.125 的 594/419。这说明数值筛查、argmax 一致和逐位一致是不同条件，不能相互替代。

## 只读离线分析

`benchmarks/analyze_fixed_path.py` 读取保存的全量 BF16 logits，分析共同平移、中心化误差与 softmax 分布差异；没有重跑 GPU，没有增加新验收阈值，也没有改原结果。

共同平移部分的平方误差能量比例定义为 `V * mean(delta)^2 / sum(delta^2)`。六个超界位置有 76.18%–94.66% 的误差能量来自共同平移，去掉平移后的相对 L2 为 1.34%–2.47%。softmax 对精确的统一平移不变，但真实误差还含非统一部分。

| 组 | 全部位置的最大中心化相对 L2 | 最大概率总变差 TV | 最大 KL(cold∥warm) |
| --- | ---: | ---: | ---: |
| regression_2048_c4 | 2.293% | 0.04913 | 0.006200 |
| heldout_2048_c4 | 2.100% | 0.02780 | 0.001929 |
| heldout_8192_c1 | 2.473% | 0.04022 | 0.004029 |

TV 定义为 `0.5 * sum(abs(p-q))`，KL 使用自然对数。全体位置 TV 最高 4.91%，不能把全部差异称为无影响的公共偏移；也不能事后只使用中心化误差把协议 v1 改判通过。

## 复跑与工件

输出目录必须不存在。原始 BF16 logits 保存为 NumPy int16 **位表示**，用 `decode_bf16()` 解码；不能把归档中的 int16 值当作数值 logits。

```bash
# 新的固定路径与真实 KV 检查；不重复旧性能矩阵
.venv/bin/python -m benchmarks.fixed_path_probe \
  --input-run results/20261117_phase1_smoke_eager --output results/new_fixed_path

# 只读分析已有原始 logits
.venv/bin/python -m benchmarks.analyze_fixed_path \
  --input-run results/20261118_fixed_path_probe --output results/new_fixed_path_analysis

# 本轮 5 个辅助测试；不启动 GPU，不重复原 8 项引擎测试
.venv/bin/python -m unittest discover -s tests -p 'test_correctness_protocol.py' -v
```

- [核验状态](../results/20261118_fixed_path_probe/status.json)、[逐组汇总](../results/20261118_fixed_path_probe/summary.json) 保留 `success=false`、`exact_invariants_passed=true`、`numerical_screen_passed=false`。
- 同目录 `inputs.json`、各组 `*_path.json`、`positions.jsonl`、`comparisons.jsonl`、`prefix_snapshots.jsonl`、`roundtrip.jsonl`、`*_logits_bf16.npz` 保存复核所需原始数据；环境、命令、diff、源码快照也在该目录。
- [离线汇总](../results/20261118_fixed_path_analysis/summary.json) 与 `positions.jsonl` 保存全部位置的补充诊断，不覆盖旧文件。
- [新测试输出](../results/20261118_protocol_analysis_tests.txt)：5 项通过，覆盖单 bit / signed-zero 检出、BF16 位解码、非有限/大误差拒绝、近并列分类、共同平移的概率不变性。原 8 项测试源码与上轮哈希一致，结果直接复用。

## 下一步

先围绕已归档的 6 个超界位置做独立数值参照：在相同 query/KV 数据下比较 paged 与 dense attention、batch/Chunked Prefill 形状变化，区分算子数值差异与缓存语义问题，并用独立校准/评估输入制定可解释的容差。不能只按本轮最大误差扩大阈值。

之后再做 CUDA Graph、128 输出、多 seed 和 GPU 历史工作集容量压力对照；这些尚未开始。已有性能工件继续复用，CPU cache 仍不进入本轮范围。
