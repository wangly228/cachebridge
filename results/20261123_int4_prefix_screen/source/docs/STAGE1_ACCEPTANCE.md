# 阶段 1 验收记录 · 2026-11-18

**验收已完成。36 块配置的原生 GPU 基线 A 通过并冻结。跨形状 BF16 数值筛查 v1 未通过，因此阶段 1 全项通过的判定为否。** 同形状对照和上游重放证明，所测冷、热路径的差异是基线已有行为。[机器判定](../results/20261118_stage1_acceptance/decision.json)

| 验收项 | 判定 | 证据 |
| --- | --- | --- |
| 压力基线 A | 通过，冻结 36 块配置 | 三个 seed，共 96 请求。24 对相同输入的引擎内 p95 TTFT 为热命中 85.2 ms、淘汰重算 1240.0 ms。[分 seed 结果](../results/20261118_pressure36/summary.json) |
| 基线与固定上游的一致性 | 通过 | 逐位置冷/热各 144 行完整词表 logits 全部逐位相同。独立重放压力 trace 的 96 条请求，每条 128 个 greedy token 均与已归档 A 相同。[上游对照](../results/20261118_accept_upstream/summary.json) |
| KV 复制与复用 | 通过已测用例 | 87 个真实模型块逐位往返一致，864 次前缀检查通过。32 块一次性 pinned 分配失败记录保留，有界 staging 路径通过。[固定路径报告](FIXED_PATH_REPORT.md) · [搬运成本](STAGE1_REPORT.md) |
| 指标采集开销 | 已测 | 6 对 trace，开启和关闭各 96 条请求。同一负载累计引擎墙钟由 70.897 s 增至 71.101 s，实测 +0.287%。完整输出相同。[逐对记录](../results/20261118_accept_metrics/pairs.jsonl) |
| 冷/热跨形状筛查 v1 | 未通过 | 144 对中有 6 对超出预设数值阈值，上游逐位复现。压力负载中，24 对热命中和淘汰重算的请求有 10 对后续 greedy 输出分叉。[v1 判据](CORRECTNESS_PROTOCOL.md) · [压力 trace 分歧](../results/20261118_pressure36/greedy_differences.json) |

## 首次观察到的分歧

在涉及这 6 个超界位置的三个请求上，对齐相同输入及完整的最后 256 个 prompt token，逐层比较 cold 与 warm prefill。下表的 token 位置和层号从 0 开始。相对 L2 按整个尾块张量计算。[逐层原始差异](../results/20261118_accept_prefill/layer_differences.jsonl)

| 组 / 请求 | 第一个出现差异的观测点 | 首个绝对 token 位置 | 该点相对 L2 | 首次出现不同 KV 的层 |
| --- | --- | ---: | ---: | ---: |
| 2048/C=4 旧回归组，请求 2 | 第 0 层 attention 输出 | 1792 | 0.0119% | 第 1 层 |
| 2048/C=4 独立组，请求 0 | 第 0 层 attention 输出 | 1792 | 0.0197% | 第 1 层 |
| 8192/C=1 独立组，请求 0 | 第 0 层 QKV 投影输出 | 7936 | 0.3169% | 第 0 层 |

两组 2048 的第 0 层 embedding、归一化输入、QKV 投影输出及写入的 K/V 均逐位一致。8192 组的归一化输入也逐位一致，最后一次 prefill 的投影输入行数由冷算的 2048 变为热算的 256。记录给出了首次出现差异的模块边界，具体 CUDA kernel 内的指令级成因未测。[调度形状](../results/20261118_accept_prefill/shapes.jsonl)

在 eager 固定 token 路径下，先将原 warm 页映射到独立临时缓冲区作为对照，144/144 行 logits 逐位复现原 warm。随后只在 decode attention 的读缓冲区换入冷算得到的最后一个 prompt KV 块，各请求继续使用同一条历史 token 路径。全部 **135/135 个 decode 位置**逐位恢复为 cold logits，包含 6 个原有超界点。其余 9 个位置是 prefill 产出的首 token，读取替换发生在其后。这个干预把这 6 个位置的差异定位到重新计算出的尾块 KV。对 36 块 Graph 压力 trace 的 10 对分叉，上游已复现，逐条干预未测。[诊断记录](../results/20261118_accept_prefill/summary.json)

## 开销和基线适用范围

压力配置固定为本地 RTX 3090、Qwen3-4B BF16、TP=1、36 个 256-token 块、并发 4、2048-token 前缀加 128-token suffix、固定输出 128，decode 开启 CUDA Graph。模型 revision、实际分配、输入、到达顺序、代码快照随基线冻结。p95 每条件 24 个样本，来自六波请求。三个 seed 的 p95 取值范围为热命中 80.3–85.2 ms、淘汰重算 1174.8–1252.2 ms。[配置](../results/20261118_pressure36/engine.json) · [环境与模型](../results/20261118_pressure36/environment.json)

开销实验交替切换 metrics，24 对请求波组成 6 对完整 trace。两侧均使用相同的外部 step 返回观察点和波结束同步。以各波 enqueue 到完成的墙钟之和计算，6 对 trace 的相对差值范围为 −0.368% 至 +0.642%。这是本次观测范围。+0.287% 对应整波耗时，正式基线的逐请求 TTFT 取自真实 token append。[测量定义](../results/20261118_accept_metrics/design.json) · [原始波数据](../results/20261118_accept_metrics/waves.jsonl)

上游参照从本地 Git 对象导出固定提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，使用独立 Python 进程并检查包导入路径。测试适配器限定相同实际 KV 块数，并在输出端使用显式 argmax 或已归档 token 路径。固定路径对比采样前的 logits，压力 trace 对比自由生成的完整 token 序列。该对照提供输出一致性证据，性能数值取自基线 A 的原测量。[上游来源与适配说明](../results/20261118_accept_upstream/reference.json)

## 复核与复跑

在仓库根目录运行。原验收清单有 474 项。公开清单移除了 10 份个人交接文档，第一条命令校验其余 464 个公开文件。第二条从原始记录和冻结的引擎快照重新汇总，适用于已开始阶段 2 开发的工作区。验收当前源码与冻结引擎的一致性时，省略 `--archived-engine`。后三条为阶段 1 源码的实验命令，输出目录须为新路径。

```bash
sha256sum --quiet -c results/20261118_stage1_acceptance/SHA256SUMS
.venv/bin/python -m benchmarks.audit_stage1 --archived-engine --output results/new_stage1_audit
.venv/bin/python -m benchmarks.metrics_overhead --output results/new_metrics_overhead
.venv/bin/python -m benchmarks.prefill_acceptance --output results/new_prefill_acceptance
.venv/bin/python -m benchmarks.upstream_acceptance --output results/new_upstream_acceptance
```

冻结的比较对象是上述配置的基线 A。未来对照可直接复用其 trace、成本表和原始计时。v1 的 6 个失败保留在验收记录中。若将跨形状数值一致性作为新的发布要求，需要独立校准和评估数据。CPU 缓存收益未测。
