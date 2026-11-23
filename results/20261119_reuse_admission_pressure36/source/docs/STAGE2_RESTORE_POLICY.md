# 阶段 2 · 成本感知恢复策略 C

**策略 C 已实现并通过正确性验证。冻结 trace 上，24 条 CPU 命中请求与 B 作出了相同的恢复选择。独立混合负载上，C 对未覆盖的长度选择重算，拉高了同一波请求的首 token 延迟。** [逐请求决策和预测误差](../results/20261118_stage2_c_report/decisions.jsonl) · [A/B/C 原始记录汇总](../results/20261118_stage2_c_report/summary.json)

## 成本表怎样进入引擎

启用 C 时在 `LLM(...)` 同时传入 `cpu_prefix_cache_blocks=64` 和 `cpu_restore_cost_table="results/20261118_stage2_c_calibration/cost_table.json"`。仅指定 CPU 容量时使用 B。两者使用同一套 CPU slot、LRU、同步搬运及重算后写回代码。C 在调度器确定 prefill 长度和占用 GPU 块之前，比较恢复和重算两个选择。[成本选择](../nanovllm/engine/restore_policy.py) · [可恢复前缀与容量判断](../nanovllm/engine/cpu_prefix_cache.py) · [调度入口](../nanovllm/engine/scheduler.py)

- 成本表绑定模型与权重 namespace、设备型号、dtype 所在模型配置、实际 GPU/CPU 容量、调度预算、CUDA Graph 开关及 Torch/CUDA/FlashAttention 版本。加载时校验这些字段和样本数、分位数。请求按连续 GPU 前缀、可恢复 CPU tokens、其后的 GPU 命中、剩余 query 长度、在引擎中的请求数和 running 请求数精确匹配成本桶。[本机成本表](../results/20261118_stage2_c_calibration/cost_table.json)
- 两条路径都预测从 enqueue 到首个真实输出 token 的延迟。用桶内 p50 作点预测，只有恢复 p90 小于重算 p10 才恢复。未测形状、成本区间重叠或抢占后重入时，保留开头的连续 GPU 命中并重算其余部分。如果放弃后面的共享 GPU 页会使重算无法分配，保留能够分配的整段恢复。逐请求记录原因、两条路径的预测、实际搬运和计算量。未测形状的预测字段留空。[请求指标](../nanovllm/engine/metrics.py) · [策略测试](../tests/test_restore_policy.py)

## 独立校准

RTX 3090、Qwen3-4B BF16、TP=1，36 个 GPU 块与 64 个 CPU slot，256 tokens/块。独立种子 701–704 共 20 对请求波、44 对请求。每一对使用完全相同的 token IDs。两臂分别从冷状态填充 CPU，再用四条互异的 2176-token 请求占满 GPU 池。核对 GPU 块映射、空闲顺序、CPU slot 和 LRU 顺序相同后，分别强制恢复、强制重算。两臂保留完整写回和查表开销，执行次序按 seed 交替。校准输出为 2 tokens，逐波核对了所有首 token 均早于第一步 decode。正式评估输出为 128 tokens。[实验设计](../results/20261118_stage2_c_calibration/design.json) · [逐对成本](../results/20261118_stage2_c_calibration/pairs.jsonl)

| 前缀 tokens | 并发 | 分排队位置的恢复 p50 (ms) | 分排队位置的重算 p50 (ms) | 成本表选择 |
| ---: | ---: | ---: | ---: | --- |
| 512 | 1 | 50.4 | 101.8 | 恢复 |
| 512 | 4 | 102.5 | 243.9–361.2 | 四个位置均恢复 |
| 2048 | 1 | 72.1 | 331.6 | 恢复 |
| 2048 | 4 | 205.9 | 333.1–1328.2 | 四个位置均恢复 |
| 8192 | 1 | 167.5 | 1470.7 | 恢复 |

各组 suffix 为 128 tokens。每个排队位置有 4 对独立种子的测量，表中范围来自不同位置。11 个桶均满足恢复 p90 小于重算 p10。该实验未测得重算更快的分界点。随后冻结成本表，SHA256 为 `f66c529cec9aad0f80666f842600c4137ab2837b4407985e8a0a2bbc826ff3c4`。[机器判定](../results/20261118_stage2_c_calibration/status.json)

## 冻结 trace 的 A/B/C 对照

复用已归档的 A、B 计时及原始输入，新增 C 实测。三组使用相同的四波顺序、实际 GPU 块数、模型、greedy、128-token 输出及 CUDA Graph。三个 seed 共 6 个冷启动周期，正式计时关闭 profiler。每个场景的分 seed p95 含 8 个相关请求，范围是三个 seed 的观测范围。[引擎配置与输入哈希](../results/20261118_stage2_c_pressure36/engine.json) · [逐请求原始时间](../results/20261118_stage2_c_pressure36/requests.jsonl)

| 引擎内 p95 TTFT | 每策略请求数 | A (ms) | B (ms) | C (ms) | C 各 seed p95 范围 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冷填充 | 24 | 1240.65 | 1391.97 | 1338.65 | 1311.87–1349.22 |
| GPU 热命中 | 24 | 85.17 | 86.66 | 85.15 | 80.82–85.16 |
| 无复用干扰 | 24 | 1237.89 | 1438.86 | 1356.27 | 1331.76–1512.30 |
| GPU 淘汰后重放 | 24 | 1239.99 | 248.64 | 211.07 | 209.58–211.08 |
| 全部请求 | 96 | 1240.37 | 1404.63 | 1335.31 | 1317.57–1347.12 |

C 的淘汰重放 p95 比 A 低 82.98%，整条 trace 的 p95 比 A 高 7.65%。C 恢复 192 块（6.75 GiB），每条重放实际只计算 128 tokens。全程写入 384 块（13.5 GiB），无复用干扰段 p95 比 A 高 9.56%。6 个冷启动周期的合计墙钟 A/B/C 分别为 71.775/69.829/67.442 s，整段输出吞吐分别为 171.20/175.97/182.20 tokens/s。三组均计入各自的缓存维护和等待。[分场景指标及内存](../results/20261118_stage2_c_report/summary.json)

**B 与 C 的选路增益。** 冻结 trace 上两组采用相同路径，C 的 96 条完整 greedy 输出均与 B 相同。另在同一引擎中按 BC/CB 交替了 6 对独立请求波，每臂 24 条请求，逐对使用相同输入和缓存状态，2048-token 前缀均恢复。每对以 4 个请求计算波内 p95，C−B 的中位差为 **+0.50 ms**，范围 −0.64 至 +8.33 ms，C 在 6 对中有 2 对更快。该对照未显示 C 优于 B 的选路收益。此前分时实测中 248.64 与 211.07 ms 差距的具体成因未定位。[交替对照原始波对](../results/20261118_stage2_c_paired_bc/pairs.jsonl) · [交替对照判定](../results/20261118_stage2_c_paired_bc/status.json)

## 独立的短、长与混合前缀

种子 901–903 与校准、冻结评估均独立。每个场景从空缓存出发，先填充，再用四条无复用请求占满 GPU，最后重放目标。每策略 90 条正式请求，其中 27 条为淘汰后重放。混合波按 512、1024、2048、512 tokens 顺序到达。1024-token 前缀在实验设计中用于检验缺少测量数据时的选择。三个策略读取完全相同的输入和到达顺序。[固定输入](../results/20261118_stage2_c_holdout/trace.json) · [分组及内存数据](../results/20261118_stage2_c_report/summary.json)

| 引擎内 p95 TTFT | 每策略请求数 | A (ms) | B (ms) | C (ms) | C 各 seed p95 范围 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512-token 前缀重放，并发 4 | 12 | 337.88 | 101.24 | 108.88 | 106.16–108.88 |
| 8192-token 前缀重放，并发 1 | 3 | 1359.73 | 173.25 | 189.69 | 184.23–190.01 |
| 混合前缀重放，并发 4 | 12 | 614.26 | 138.09 | 290.69 | 285.48–290.70 |
| 全部重放 | 27 | 1350.86 | 165.73 | 290.69 | 285.48–290.69 |
| 含填充和干扰的全部请求 | 90 | 1342.41 | 1334.19 | 1352.06 | 1345.25–1359.13 |

混合重放中，B 恢复 48 块，实际计算 1536 tokens。C 恢复 36 块，另有 3 条 CPU 已命中的请求因为 1024-token 桶缺失而回退重算。C 实际计算 4608 tokens，并额外写回 12 块，混合 p95 比 B 高 **110.51%**。这些同波请求共享调度等待。C 在冻结集 24 次有预测的决策上，所选路径 TTFT 的平均绝对误差为 3.17 ms。在混合波有预测的 9 条请求上，误差为 144.88 ms。当前分桶没有描述同波其他请求的长度及其回退工作量。8192-token 重放每策略仅 3 条，表中范围为每个种子的单次观察。[逐请求决策](../results/20261118_stage2_c_report/decisions.jsonl)

**补测记录。** 首次在同一进程顺序加载三组引擎时，B 正式波的 GPU 分配峰值为 16.58 GiB，混合重放 p95 为 146.86 ms。已保留该次记录。只将 B 在独立进程中补测，核对输入 trace、成本表、模型、依赖和引擎源码后，采用补测值 138.09 ms。最终 A/B/C 的实际 GPU 池均为 36 块，GPU 分配峰值分别为 9.01/9.04/9.04 GiB。复跑脚本现对每个策略使用独立进程。[首次 B 波记录](../results/20261118_stage2_c_holdout/B/waves.jsonl) · [独立进程 B 波记录](../results/20261118_stage2_c_holdout_b_isolated/B/waves.jsonl)

**正确性。** 冻结集 96/96、独立集 90/90 条 C 请求的 128 个 greedy 输出均通过同形状对照。对照混合重放的 12 条请求时，关闭成本选择，在 B 的 CPU 索引中仅隐藏被拒绝前缀的第一块，其余前缀照常恢复。逐条核对输入、实际恢复量、实际计算量及全部输出。正式 C 运行的抢占和传输失败数均为 0。25 项单元测试通过，覆盖未测形状、成本区间重叠、重算更便宜的选择、共享页容量约束、抢占和复制故障。此次 B/C 共用已验过的 KV 搬运实现。[同形状对照](../results/20261118_stage2_c_holdout/mixed_reference/checks.jsonl) · [单元测试](../results/20261118_stage2_c_tests.txt) · [B 的 KV 和完整 logits 验证](STAGE2_CPU_CACHE.md)

## 复跑与下一步

在仓库根目录使用现有环境。以下依次生成新的成本表、冻结 trace 的 C 数据、独立 A/B/C 数据、同进程交替对照和离线汇总。

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/pegaflow-inductor
export TRITON_CACHE_DIR=/tmp/pegaflow-triton
.venv/bin/python -m benchmarks.restore_calibration --output results/new_c_cal
.venv/bin/python -m benchmarks.cpu_cache_pressure --cost-table results/new_c_cal/cost_table.json --output results/new_c_frozen
.venv/bin/python -m benchmarks.restore_holdout --cost-table results/new_c_cal/cost_table.json --output results/new_c_holdout
.venv/bin/python -m benchmarks.restore_bc_control --cost-table results/new_c_cal/cost_table.json --output results/new_c_control
.venv/bin/python -m benchmarks.compare_restore_policy --calibration results/new_c_cal --frozen-c results/new_c_frozen --holdout results/new_c_holdout --output results/new_c_report
```

只复核本次已归档的数据时，可直接运行离线汇总。

```bash
.venv/bin/python -m benchmarks.compare_restore_policy \
  --calibration results/20261118_stage2_c_calibration --frozen-c results/20261118_stage2_c_pressure36 \
  --holdout results/20261118_stage2_c_holdout --holdout-b results/20261118_stage2_c_holdout_b_isolated \
  --output /tmp/pegaflow_c_report
```

2026-11-18 后续实验已用新种子补测 1024-token 与混合队列的恢复/重算成本，并在同一引擎中轮换 B、此处的原版 C 和新版 C。48 条独立重放的回退从 12 次降为 0 次，p95 从 276.70 降为 138.31 ms。同步写入的无复用开销、独立预测误差和整段 p95 均已记录。[下一轮实测与复跑](STAGE2_QUEUE_CALIBRATION.md)
