# 阶段 2：prefill 入批边界与恢复选路

2026-11-18 在本地 RTX 3090 执行，11-19 完成核验。**按当前步可接纳的工作量选路（C3），能更准确地预测目标请求的首 token，但在本轮负载上应采用 CPU 命中即恢复的 B。** 6 个独立种子、每策略 96 条重放，p95 TTFT 分别为 B 336.75、上一版队列策略 C2 338.72、C3 455.48 ms。C3 比 B 高 35.26%，每个种子的重放 p95 都更高。24 波中共改选 6 次，每次目标更快，每次整波 p95 更高。[分组与配对统计](../results/20261118_step_c_report/summary.json) · [逐波选择与两条路径](../results/20261118_step_c_report/cases.jsonl)

## 从哪里继续

上一轮已完成 1024-token 和混合队列校准、6 种子交替对照和冷同伴探测。本轮先核对三个运行的成功判定、冻结表 SHA256 和引擎源码快照。上一轮 640、2176 tokens 的 8 对直接从逐步、逐请求日志恢复。新增校准和独立评估均使用新的输入。[上一轮结果](STAGE2_QUEUE_CALIBRATION.md) · [复用的 8 对](../results/20261118_queue_c_peer_probe/pairs.jsonl)

Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`，BF16，单卡 RTX 3090。GPU 实配 36 块、CPU 64 块，每块 256 tokens / 36 MiB。并发 4，每步最多计算 2048 tokens，decode 启用 CUDA Graph。每个周期从空缓存填充，再用 4 条各 2176 tokens 的互异请求占满 GPU，最后重放。重放前 3 条分别有 512、1024、512 tokens 的 CPU 前缀，各有 128 tokens 待算。第 2 条是选路目标。第 4 条为全冷请求，同一种子下只改变这条输入的截取长度。[引擎实配](../results/20261118_step_c_evaluation/engine.json) · [模型与源码快照](../results/20261118_step_c_evaluation/environment.json) · [固定输入](../results/20261118_step_c_evaluation/trace.json)

## 哪个请求能进入当前步

恢复目标时，前三条共算 384 tokens，留给冷请求 **1664**。目标重算时，前三条共算 1408 tokens，留给冷请求 **640**。调度器按队首顺序接纳，余量恰好等于请求长度时可以入批。下表每行 4 对，每格为目标从 enqueue 到首个真实输出 token 的中位数。新增 6 个长度、24 对用种子 1401–1404，表中的 640 和 2176 两行来自种子 1301–1304。两条路径的其他请求选择固定，重放前缓存状态相同。[32 对原始成本](../results/20261118_step_c_calibration/pairs.jsonl) · [校准设计](../results/20261118_step_c_calibration/design.json)

| 冷请求长度 | 冷请求与目标同一步：恢复 / 重算 | 目标恢复 ms | 目标重算 ms |
| ---: | --- | ---: | ---: |
| 512 | 是 / 是 | 155.31 | 274.66 |
| 639 | 是 / 是 | 166.34 | 283.01 |
| 640 | 是 / 是 | 163.71 | 279.33 |
| 641 | 是 / 否 | 195.00 | 224.06 |
| 1663 | 是 / 否 | 290.18 | 224.39 |
| 1664 | 是 / 否 | 288.98 | 225.25 |
| 1665 | 否 / 否 | 89.99 | 224.38 |
| 2176 | 否 / 否 | 88.39 | 220.01 |

C3 复用 C2 的 24 个桶，在相同目标形状下分别估算恢复和重算能接纳的序列数、实算 query tokens 和 H2D tokens。11 个测量点来自上述 32 对。相同动作、序列数和 H2D 量下，只在已测 query 范围内插值。多个冷长度得到相同步时，先取每个种子的中位数，再计算跨种子分位数。仍按恢复 p90 严格低于重算 p10 选恢复。两条路径缺少测量、目标无法在本步完成、先前运行请求不属于本步或 GPU 无法保证分配时，沿用 C2 查表。推算只读缓存元数据，后续同伴按连续可恢复前缀估算。本轮校准限定在上述队列位置和目标形状。[冻结的 C3 表](../results/20261118_step_c_calibration/cost_table.json) · [调度器](../nanovllm/engine/scheduler.py) · [选路逻辑](../nanovllm/engine/restore_policy.py)

## 独立评估

种子 1501–1506，正式冷请求长度为 608、704、1600、1728，均未用于校准。每个长度在同一引擎中轮换 B、C2、C3 的六种排列。三组输入和各波初态相同，每策略 24 个周期、288 条正式请求，每条 greedy 输出 128 tokens，忽略 EOS。TTFT 含排队、查表、搬运与计算。整段计时从入队到各波推理和同步搬运完成，含填充和干扰。加载、编译、预热和写日志均在计时外，关闭 profiler。[评估设计](../results/20261118_step_c_evaluation/design.json) · [正式逐请求与逐步记录](../results/20261118_step_c_evaluation/measurements)

| 逐请求 p95 TTFT | 每策略样本数 | B ms | C2 ms | C3 ms |
| --- | ---: | ---: | ---: | ---: |
| 608-token 冷请求重放 | 24 | 168.15 | 168.59 | 168.92 |
| 704-token 冷请求重放 | 24 | 197.18 | 197.77 | 198.60 |
| 1600-token 冷请求重放 | 24 | 302.28 | 303.04 | 457.64 |
| 1728-token 冷请求重放 | 24 | 340.25 | 341.00 | 341.58 |
| 全部重放 | 96 | 336.75 | 338.72 | 455.48 |
| 冷填充 | 96 | 430.23 | 429.98 | 430.23 |
| 无复用干扰 | 96 | 1299.85 | 1301.46 | 1302.94 |
| 填充、干扰和重放整段 | 288 | 1284.00 | 1286.75 | 1284.78 |

**1600 tokens 是全部 6 次实际改选发生的位置。** 恢复使 4 条请求同一步执行，共算 1984 tokens。重算使前三条先算 1408 tokens，第 4 条冷请求进入下一步。按相同种子配对，C3 相对 B 的目标 TTFT 中位差为 −61.26 ms（范围 −62.89 至 −59.09），冷请求为 +154.47 ms（+153.39 至 +156.50）。每波按 4 条请求计算 p95，六对波的 p95 差中位数为 +122.18 ms（+121.05 至 +124.16）。合并四种长度，按种子分别计算重放 p95 后，C3 减 B 的范围为 +57.05 至 +60.04 ms。这些范围是种子间的观察值。上表按各行的全部请求合并计算 p95。[每个种子的两条路径和实算量](../results/20261118_step_c_report/cases.jsonl)

| 48 个相同观测上的目标 TTFT 预测误差（MAE） | 观测数 | C2 ms | C3 ms |
| --- | ---: | ---: | ---: |
| 恢复 | 24 | 67.17 | 5.66 |
| 重算 | 24 | 43.61 | 6.10 |
| 两条路径合计 | 48 | 55.39 | 5.88 |

恢复使用 B 的正式观测，重算使用改选后的 6 次正式 C3 观测，另外 18 次用相同输入和缓存状态强制重算，只生成 2 tokens。所有路径上各请求都在本波首次 decode 前拿到首 token，这 18 次只用于比较首 token 延迟。预测的当前步序列数、实算量和搬运量与全部 48 条路径的实际记录吻合。在全部 32 对校准和 24 对独立路径观测里，恢复的波内 p95 都更低。[补测原始记录](../results/20261118_step_c_evaluation/controls) · [逐波预测](../results/20261118_step_c_report/cases.jsonl)

| 24 个周期合计 | B | C2 | C3 |
| --- | ---: | ---: | ---: |
| 96 条重放的 CPU 命中 / 实际恢复请求 | 72 / 72 | 72 / 72 | 72 / 66 |
| 重放实际计算的 prompt tokens | 37056 | 37056 | 43200 |
| 整段 D2H / H2D 块数 | 1104 / 192 | 1104 / 192 | 1128 / 168 |
| 填充 / 干扰 / 重放墙钟，s | 54.566 / 78.500 / 51.435 | 54.495 / 78.550 / 51.467 | 54.533 / 78.483 / 52.383 |
| 整段墙钟，s | 184.501 | 184.513 | 185.400 |
| 整段输出吞吐，tokens/s | 199.80 | 199.79 | 198.84 |

三策略在重放段的 GPU 首次命中都为 0，CPU 首次命中均为 49152 tokens。C3 多算了 6144 个前缀 token，整段写入 39.66 GiB、读回 5.91 GiB。三组的抢占、保存失败、恢复失败均为 0。GPU 峰值 allocated / reserved 为 9.04 / 9.18 GiB，波末设备占用的最高样本为 10.51 GiB，共用进程的 CPU 峰值 RSS 为 4.65 GiB。[延迟、吞吐、搬运及内存](../results/20261118_step_c_report/summary.json)

**正确性。** C2 和 C3 全部 576 条正式请求的 128 个 greedy 输出通过同形状核对。对改选波另用关闭成本选择的 B 隐藏目标 CPU 前缀，独立重放 24 条，逐条核对输入、输出、恢复量与实际计算量。32 项单元测试通过，含预算等于边界、队首阻塞、容量不足、插值区间外沿用原策略等回归。[评估判定](../results/20261118_step_c_evaluation/status.json) · [独立输出对照](../results/20261118_step_c_evaluation/reference) · [单元测试](../results/20261118_step_c_tests.txt)

## 复核与交接

只复核归档时，在仓库根目录运行离线汇总。输出目录须使用新路径。

```bash
.venv/bin/python -m benchmarks.step_report \
  --calibration results/20261118_step_c_calibration \
  --evaluation results/20261118_step_c_evaluation \
  --output /tmp/pegaflow_step_review
```

完整复跑沿用现有环境和上一轮 8 对记录。校准与正式评估依次执行。

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/pegaflow-inductor
export TRITON_CACHE_DIR=/tmp/pegaflow-triton
.venv/bin/python -m benchmarks.step_calibration --output results/new_step_cal
.venv/bin/python -m benchmarks.step_evaluation --calibration results/new_step_cal --output results/new_step_eval
.venv/bin/python -m unittest discover -s tests -v
```

评估按“种子、冷请求长度、策略”保存完成记录。中断后，在同一条评估命令末尾加 `--resume` 可继续未完成的对照。C3 由显式配置 `cpu_restore_cost_table="results/20261118_step_c_calibration/cost_table.json"` 启用，CPU 容量设为 64 块。已核验 72 个正式运行记录、366 个原始文件哈希及正式评估的 23 个引擎源码文件。[校准工件](../results/20261118_step_c_calibration) · [评估源码](../results/20261118_step_c_evaluation/source) · [原始文件 SHA256](../results/20261118_step_c_report/raw_sha256.json) · [汇总命令与核验清单](../results/20261118_step_c_report/provenance.json)

本轮完成两个入批边界的验证。对这组负载，B 保留为重放 p95 的性能参照，C3 保留为目标延迟与整波尾延迟发生冲突的复现实例。2026-11-19 的总验收已将该结果与冻结负载的淘汰重放、无复用及整段数据一并归档。[阶段 2 验收报告](STAGE2_ACCEPTANCE.md)
