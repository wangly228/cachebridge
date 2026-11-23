# 阶段 2 实验：1024-token 前缀与混合队列的恢复成本

2026-11-18，本地 RTX 3090，Qwen3-4B BF16。冻结成本表后，用 6 个独立种子对比命中即恢复的 B、原版 C 和新版 C。**在两种混合队列的 48 条淘汰重放请求上，因缺少测量而回退重算的次数从 12 降到 0，p95 TTFT 从 276.70 降到 138.31 ms（−50.01%），6 个种子全部改善。** B 的 p95 为 137.09 ms，B 和新版 C 在 48 条请求上都选择了恢复。含填充与无复用干扰的 144 条请求，原版和新版 C 的 p95 分别为 1310.91 和 1308.37 ms（−0.19%）。[机器汇总](../results/20261118_queue_c_report/summary.json) · [逐请求记录](../results/20261118_queue_c_evaluation/requests.jsonl)

## 测量条件

模型 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`，权重和 KV 均为 BF16。单卡，GPU 实配 36 块，CPU 64 块，每块 256 tokens / 36 MiB。每步 prefill 上限为 2048 tokens，引擎上限 8 条序列，正式每波同时入队 4 条。decode 启用 CUDA Graph，prefill 用 eager。正式每条请求 greedy 输出 128 tokens，忽略 EOS。校准和补充实验只输出 2 tokens，所有重放请求的首 token 均在该波第一次 decode 前产生。[引擎实配](../results/20261118_queue_c_evaluation/engine.json) · [环境与模型文件哈希](../results/20261118_queue_c_evaluation/environment.json)

每个周期从空缓存开始。先填充目标前缀，再让 4 条互不复用的 2176-token 请求实际占满 36 个 GPU 块，最后重放目标前缀并更换 128-token 尾部。每次对照复用完全相同的输入 IDs，并核对重放前的 GPU 空闲页、前缀索引、CPU slot 和 LRU 顺序。TTFT 从各自 enqueue 计到首个真实输出 token。整段墙钟取三波从入队到全部输出和搬运完成的耗时之和，包含 D2H、H2D、查表、等待、prefill 和 decode。加载、capture、预热、写盘、正确性参照均在正式时间之外，关闭 profiler。[输入 trace](../results/20261118_queue_c_evaluation/trace.json) · [配对初态](../results/20261118_queue_c_evaluation/matched_states.jsonl) · [逐步记录](../results/20261118_queue_c_evaluation/steps.jsonl)

## 路径成本与策略实现

种子 1101–1104 用于校准，1100 用于预热。每个种子、每个队列先后执行全员恢复和逐一指定目标重算的实验。只改变目标的选择，其他请求照常恢复。实验顺序在相邻种子间反转，同一波全员恢复记录由各目标共享。共有 52 对目标观测，每个位置 4 对。下表为目标请求端到端 TTFT 的中位数。[52 对原始记录](../results/20261118_queue_c_calibration/pairs.jsonl) · [校准设计](../results/20261118_queue_c_calibration/design.json)

| 1024-token 目标所在队列 | 目标位置 | 位置样本量 | 恢复 ms | 重算 ms |
| --- | --- | ---: | ---: | ---: |
| 单请求 | 第 1 条 | 4 对 | 55.50 | 178.60 |
| 4 条均为 1024 | 第 1 至第 4 条分别校准 | 各 4 对 | 136.14–136.16 | 274.19–275.12 |
| 512 / 1024 / 2048 / 512 | 第 2 条 | 4 对 | 136.57 | 276.46 |
| 512 / 1024 / 512 / 512 | 第 2 条 | 4 对 | 110.24 | 251.28 |

后两行中，其他三条各有 128 个待算 token，合计待恢复的前缀分别是 3072 和 1536 tokens。让目标重算后，其他三条的 TTFT 配对差中位数分别为 +138.60 和 +140.81 ms，每组有 4 波、12 个同伴观测。52 对中有 4 对在重算路径上因批预算推迟到后续步骤。两条路径按目标第一次可以做决策时的同一队列状态配对，同时记录最终分配时的状态。[逐对成本及同伴延迟变化](../results/20261118_queue_c_calibration/pairs.jsonl)

原版表的 11 个成本桶保留。新版加入 5 个 1024-token 基本形状桶和 8 个混合队列桶，总计 24 个。调度器先按目标长度、尾部、队列位置及其他请求的待算和待恢复 token 数查实测值，再查不含同伴信息的基本形状。两层都没有数据时回退重算。恢复条件仍为恢复 p90 严格小于重算 p10。本轮新增的 13 个桶全部满足恢复条件。每桶只有 4 对观测，分位数为经验值。[冻结成本表](../results/20261118_queue_c_calibration/cost_table.json) · [表版本与哈希](../results/20261118_queue_c_calibration/status.json)

同一步已接纳的同伴按实际安排的 query 和恢复量计数。后续等待请求只在剩余序列名额内查看，按当前连续前缀可全部恢复来估计。查找只读元数据，遇到有队列成本桶的候选才查看同伴。同步保存、恢复、LRU 和容量约束沿用 B。相同配置下使用 `cpu_prefix_cache_blocks=64` 和 `cpu_restore_cost_table="results/20261118_queue_c_calibration/cost_table.json"` 即启用新版。[调度器](../nanovllm/engine/scheduler.py) · [缓存块选择](../nanovllm/engine/cpu_prefix_cache.py) · [成本决策](../nanovllm/engine/restore_policy.py)

## 独立性能对照

种子 1201–1206 用于评估，1200 用于预热。每种布局在同一引擎中各执行 B、原版 C、新版 C 的六种排列各一次。第二种布局将 1024-token 请求从校准时的第 2 位移到第 3 位。评估前已冻结成本表，评估请求的首块与本轮校准及上一轮策略 C 的校准、独立评估输入均无交集。每策略 144 条正式请求，共 432 条。下表为逐请求 p95 TTFT。[评估设计](../results/20261118_queue_c_evaluation/design.json) · [原始波耗时](../results/20261118_queue_c_evaluation/waves.jsonl)

| 场景 | 每策略请求数 | B ms | 原版 C ms | 新版 C ms |
| --- | ---: | ---: | ---: | ---: |
| 512 / 1024 / 2048 / 512 重放 | 24 | 137.09 | 277.29 | 138.32 |
| 512 / 512 / 1024 / 512 重放 | 24 | 111.36 | 249.88 | 111.31 |
| 两种布局合计重放 | 48 | 137.09 | 276.70 | 138.31 |
| 从空缓存填充 | 48 | 655.83 | 657.20 | 654.93 |
| 无复用干扰 | 48 | 1314.61 | 1319.04 | 1317.17 |
| 三段全部请求 | 144 | 1308.85 | 1310.91 | 1308.37 |

原混合布局的重放 p95 相对原版 C 降低 50.12%，换序布局降低 55.46%。按种子配对，两种布局合计重放的新版减原版 p95 差值为 −141.60 至 −132.73 ms，中位数 −137.54 ms。新版减 B 为 −0.79 至 +2.78 ms，中位数 +0.65 ms，仅 2/6 个种子更快。整段三类请求上，新版仅在 2/6 个种子比原版 C 更快。范围表示六次配对观察，波内请求共享等待。[各种子 p95、配对差及分布](../results/20261118_queue_c_report/summary.json)

| 两种布局合计 | B | 原版 C | 新版 C |
| --- | ---: | ---: | ---: |
| 重放中 CPU 命中 / 实际恢复请求 | 48 / 48 | 48 / 36 | 48 / 48 |
| 重放实际计算 prompt tokens | 6144 | 18432 | 6144 |
| 整段 D2H / H2D 块数 | 540 / 156 | 588 / 108 | 540 / 156 |
| 填充 / 干扰 / 重放墙钟，s | 29.847 / 39.946 / 24.782 | 29.874 / 40.025 / 26.454 | 29.892 / 40.006 / 24.819 |
| 整段墙钟，s | 94.574 | 96.352 | 94.717 |
| 整段输出吞吐，tokens/s | 194.89 | 191.30 | 194.60 |

重放段三组的初始 GPU 命中均为 0，CPU 均命中 39936 tokens。新版省去原版 C 的 12288 个前缀重算 token 和 48 块额外写回。新版整段写入 18.98 GiB、读回 5.48 GiB。三策略抢占、保存及恢复失败均为 0。GPU 峰值 allocated 为 9.04 GiB、reserved 为 9.22 GiB，波末设备占用的最大样本为 10.55 GiB。CPU store 容量 2.25 GiB，共用进程的峰值 RSS 为 4.64 GiB。C 的 288 条请求均通过全部 128 个 greedy token 的同形状对照。原版 C 有 48 条重放请求额外用关闭成本选择的 B 复现实际选择，输入、恢复量、计算量和输出全部一致。[正确性判定](../results/20261118_queue_c_evaluation/status.json) · [原布局参照](../results/20261118_queue_c_evaluation/reference/C1/mixed/checks.jsonl) · [换序参照](../results/20261118_queue_c_evaluation/reference/C1/lighter_permuted/checks.jsonl)

## 队列信息改善了哪些预测

新版在 48 次决策中有 36 次命中队列桶、12 次采用基本形状桶。只补 5 个基本形状桶的版本也会在这 12 波的全部 48 条请求上恢复。用这 48 条相同实测时间离线评分，只补形状的首 token 预测平均绝对误差（MAE）为 **23.49 ms**，加入队列信息后为 **4.89 ms**。原混合布局为 34.68 → 1.08 ms，换序布局为 12.30 → 8.71 ms。只补基本形状版本的独立运行 p95 未测。本轮测出了队列信息的预测收益，重放少算的 12288 tokens 全部来自补足 1024-token 成本后改变的 12 次选择。[逐请求实际选择与离线预测](../results/20261118_queue_c_evaluation/decisions.jsonl)

在两种正式布局中，全员恢复时，任一目标的另外三个同伴合计都有 384 个待算 token。为单独改变这个量，在冻结表之后另用种子 1301–1304 做了 8 对路径测量。目标仍为第 2 条的 1024-token 前缀，两个同伴各恢复 512 tokens，最后一条冷请求使用同一串输入的前 640 或全部 2176 tokens。每个种子四种条件的初态一致。表中是从 enqueue 计起的目标 TTFT 中位数，每格 4 次。[补充实验原始配对](../results/20261118_queue_c_peer_probe/pairs.jsonl) · [逐步调度](../results/20261118_queue_c_peer_probe/steps.jsonl)

| 冷请求长度 | 同伴待算总量 | 目标与冷请求同一步执行 | 目标所在步实算 tokens，恢复 / 重算 | 目标 TTFT ms，恢复 / 重算 |
| ---: | ---: | --- | ---: | ---: |
| 640 | 896 | 两条路径均 4/4 | 1024 / 2048 | 163.71 / 279.33 |
| 2176 | 2432 | 两条路径均 0/4 | 384 / 1408 | 88.39 / 220.01 |

两组同伴待搬运总量同为 1024 tokens。长冷请求超过该步剩余预算，被推迟到后续步骤，目标反而更早得到首 token。两个新形状均采用基本桶，给出相同的恢复 / 重算预测 136.15 / 274.19 ms。只看等待队列总量无法从这两组数据推断目标所在步的工作量。[补充实验判定与预测](../results/20261118_queue_c_peer_probe/status.json)

## 复跑与下一步

在仓库根目录使用现有 `.venv`。校准依赖已归档的原版 11 桶表和 36 块 baseline 配置。下列 GPU 命令按顺序执行，输出目录取新路径。只复核已有结果时，直接运行最后一条，把校准和评估路径换为本报告中的归档目录。

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/pegaflow-inductor
export TRITON_CACHE_DIR=/tmp/pegaflow-triton
.venv/bin/python -m benchmarks.queue_calibration --output results/new_queue_cal
.venv/bin/python -m benchmarks.queue_evaluation --calibration results/new_queue_cal --output results/new_queue_eval
.venv/bin/python -m benchmarks.queue_peer_probe --calibration results/new_queue_cal --output results/new_queue_peer
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.queue_report --calibration results/new_queue_cal --evaluation results/new_queue_eval --output results/new_queue_report
```

本轮 29 项单元测试通过，覆盖旧表加载、队列匹配、未见队列的基本形状回退、查看范围和只读缓存语义。各运行目录保留命令、未提交源码快照、环境、原始输入和输出。阶段 1 与原版 C 的数据继续使用原归档。[测试日志](../results/20261118_queue_c_tests.txt) · [评估源码快照](../results/20261118_queue_c_evaluation/source) · [原版 C 报告](STAGE2_RESTORE_POLICY.md)

后续已完成 640 和 1664-token 入批边界实验。直接复用本文的 8 对冷同伴测量，新增 24 对校准。6 个独立种子上，按当前步估算的 C3 把目标预测 MAE 从 55.39 降至 5.88 ms。6 次改选都让目标请求更快，整波 p95 都更高。每策略 96 条重放的 p95 为 B 336.75、本文的 C2 338.72、C3 455.48 ms。[后续实测与复跑](STAGE2_STEP_COST.md)
