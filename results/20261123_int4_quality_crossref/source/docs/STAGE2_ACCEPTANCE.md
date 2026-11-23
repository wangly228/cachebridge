# cachebridge 阶段 2 验收报告

2026-11-19 · 验收时项目名为 PegaFlow · 单卡 RTX 3090 · Qwen3-4B BF16

**阶段 2 工程验收通过。** 在 nano-vLLM 上实现了有界同步 CPU 前缀 KV 缓存、LRU 和基于实测成本的恢复选择。冻结负载中，CPU 命中即恢复将 GPU 淘汰后重放的 p95 首 token 延迟降低 **79.95%**。同一负载的无复用段 p95 上升 **16.24%**，整段上升 **13.24%**。设计中的重放改善 20% 目标已达到，无复用退化不超过 5% 的目标未达到。[同负载逐请求汇总](../results/20261118_stage2_c_report/summary.json)

## 实现与正确性

验收后的写入优化在同一冻结输入上复测了原 B 和按复用写入的 B。新策略达到无复用和淘汰重放的两项目标。下文的历史数字对应原写入路径。[后续写入优化报告](STAGE2_WRITE_ADMISSION.md)

| 验收项 | 实现与验证 |
| --- | --- |
| 有界 CPU 存储与真实搬运 | 只保存已计算的完整 prompt 块。块管理器校验模型与配置标识、前缀哈希链和 token IDs，只复用最长连续前缀，保留末尾 query 块。Runner 在实际分层 KV 布局上打包和恢复，使用各一块 36 MiB 的锁页主存和 GPU 暂存。活跃 attention 始终从 GPU 读 KV。[块管理](../nanovllm/engine/cpu_prefix_cache.py) · [搬运实现](../nanovllm/engine/kv_transfer.py) |
| 淘汰、重复恢复与尾块刷新 | GPU 3 块、CPU 4 块的真机检查中，6 次 LRU 淘汰按预期发生。两次恢复共核对 4 个真实 KV 块及 8 行同形状完整 logits，均逐字节一致。另对 512-token 整块尾部重算后再恢复，恢复值与更新后的 GPU 版本相差 0 字节。[逐块校验](../results/20261118_stage2_b_probe/kv.jsonl) · [逐位置 logits](../results/20261118_stage2_b_probe/logits.jsonl) · [LRU 记录](../results/20261118_stage2_b_probe/status.json) · [尾块检查](../results/20261118_stage2_b_full_tail/status.json) |
| 输出与回归 | 冻结负载中，B 的 96/96 条请求与同形状参照的 128 个 greedy 输出一致。最新独立评估中，C2、C3 共 576 条正式请求通过完整输出核对，包括改选波的 24 条独立重放。归档的 32 项单元测试全过，覆盖块边界、共享引用、抢占、复制失败和入批边界。[B 的判定](../results/20261118_stage2_b_pressure36/status.json) · [C2/C3 的判定](../results/20261118_step_c_evaluation/status.json) · [改选波参照](../results/20261118_step_c_evaluation/reference) · [测试日志](../results/20261118_step_c_tests.txt) |

同形状参照区分了 KV 复制误差与 BF16 执行形状引起的差异。冻结负载的 24 条淘汰重放中，B 恢复与 A 的 GPU 热命中输出全部一致，直接与 A 在淘汰段的冷重算比较则有 14/24 一致。阶段 1 的跨形状数值筛查仍有 6/144 个位置超界，固定上游复现了全部 6 处。[同场景输出比较](../results/20261118_stage2_b_pressure36/trace_comparison.json) · [阶段 1 验收](STAGE1_ACCEPTANCE.md)

## 同一冻结负载上的 A/B/C

**A** 使用原生 GPU 前缀缓存，GPU 未命中时重算。**B** 增加 CPU 层，连续 CPU 前缀可分配时即恢复。**C1** 是首版成本策略，与 B 共用容量、LRU 和搬运路径。两条路都可分配时，C1 仅在恢复 p90 严格低于重算 p10 时选恢复。后续的 **C2** 补入 1024-token 前缀和同批请求的工作量，**C3** 分别估算两种选择能排进当前 prefill 步的工作量。引擎默认走 A，设置 `cpu_prefix_cache_blocks=64` 即启用 B，另设 `cpu_restore_cost_table` 加载对应的冻结表即可启用 C。[配置入口](../nanovllm/config.py) · [C1 成本表](../results/20261118_stage2_c_calibration/cost_table.json) · [C2 成本表](../results/20261118_queue_c_calibration/cost_table.json) · [C3 成本表](../results/20261118_step_c_calibration/cost_table.json)

模型 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。每块 256 tokens，GPU 池限为 36 块，B/C 的 CPU 池为 64 块（2.25 GiB）。并发 4，每条请求为 2048-token 前缀加 128-token 尾部，greedy 输出 128 tokens，忽略 EOS。prefill 每步最多 2048 tokens，decode 启用 CUDA Graph。种子 101–103 各运行两个空缓存周期，每周期 4 波。各策略的输入 IDs、到达次序和实际 GPU 容量一致。真实的四条干扰请求占满 GPU 后再重放目标前缀。[冻结输入](../results/20261118_pressure36/trace.json) · [A 配置](../results/20261118_pressure36/engine.json) · [B 配置](../results/20261118_stage2_b_pressure36/engine.json) · [C1 配置](../results/20261118_stage2_c_pressure36/engine.json) · [模型与依赖](../results/20261118_stage2_b_pressure36/environment.json)

**指标为引擎内 p95 TTFT，单位 ms。** 每条请求从入队计到首个真实生成 token，含排队、查表、同步搬运与计算。排除加载、编译、预热和写盘，正式计时关闭 profiler。整段 p95 从全部逐请求记录重新取分位数。

| 场景 | 每策略请求数 | A | B | C1 | B 相对 A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 空缓存填充 | 24 | 1240.65 | 1391.97 | 1338.65 | +12.20% |
| GPU 热命中 | 24 | 85.17 | 86.66 | 85.15 | +1.75% |
| 无复用干扰 | 24 | 1237.89 | 1438.86 | 1356.27 | +16.24% |
| GPU 淘汰后重放 | 24 | 1239.99 | 248.64 | 211.07 | −79.95% |
| 四段合计 | 96 | 1240.37 | 1404.63 | 1335.31 | +13.24% |

每次重放，A 实算 2176 tokens，B 从 CPU 恢复 8 块，仅计算末尾 128 tokens。B 在填充和无复用干扰中各同步写入 192 块，合计 13.5 GiB，重放读回 6.75 GiB。B 的 6 波冷填充共耗时 21.071 s。6 个完整周期墙钟 A/B/C1 分别为 71.775 / 69.829 / 67.442 s，对应整段输出吞吐为 171.20 / 175.97 / 182.20 tokens/s，含 prefill 和 decode。冻结负载中各策略抢占和搬运失败数均为 0。[A 原始请求](../results/20261118_pressure36/requests.jsonl) · [B 原始请求](../results/20261118_stage2_b_pressure36/requests.jsonl) · [C1 原始请求](../results/20261118_stage2_c_pressure36/requests.jsonl) · [B 逐波成本](../results/20261118_stage2_b_pressure36/waves.jsonl) · [整段汇总](../results/20261118_stage2_c_report/summary.json)

三个种子各含 8 条淘汰重放，种子内 p95 的观察范围分别为 A 1174.79–1252.16、B 214.29–248.66、C1 209.58–211.08 ms。上表为分时运行，B 与 C1 的 24 次恢复选择完全相同。另用同一引擎交替运行 6 对请求波（每策略 24 条），C1 减 B 的波内 p95 差中位数为 **+0.50 ms**，范围 −0.64 至 +8.33 ms。分时运行中 211.07 与 248.64 ms 差距的具体原因未定位。[分种子统计](../results/20261118_stage2_c_report/summary.json) · [交替对照原始波对](../results/20261118_stage2_c_paired_bc/pairs.jsonl)

## 混合请求下的选路

最初的独立混合负载（种子 901–903，每策略 12 条重放）已暴露 C1 的回退代价。A、B、C1 的重放 p95 分别为 614.26、138.09、290.69 ms。C1 对 3 个已在 CPU 命中但缺少成本数据的 1024-token 前缀选择了重算。此后才使用新输入校准和评估 C2。[该批输入](../results/20261118_stage2_c_holdout/trace.json) · [同批 A/B/C1 汇总](../results/20261118_stage2_c_report/summary.json)

下表各行对应同一批输入上的独立对照，单位仍为引擎内 p95 TTFT（ms）。全员 CPU 命中组使用种子 1201–1206。含冷请求组使用种子 1501–1506，每波前三条命中 CPU，第四条冷算。每组在同一引擎中轮换三个策略，整段包含空缓存填充、无复用干扰及重放。[全命中组配置与种子](../results/20261118_queue_c_evaluation/design.json) · [含冷请求组配置与种子](../results/20261118_step_c_evaluation/design.json)

| 负载与计时范围 | 每策略请求数 | B | C1 | C2 | C3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全员 CPU 命中，重放 | 48 | 137.09 | 276.70 | 138.31 | 未测 |
| 同组整段 | 144 | 1308.85 | 1310.91 | 1308.37 | 未测 |
| 含一条冷请求，重放 | 96 | 336.75 | 未测 | 338.72 | 455.48 |
| 同组整段 | 288 | 1284.00 | 未测 | 1286.75 | 1284.78 |

- **缺少成本数据会拖慢同批请求。** 全命中组中，C1 对 1024-token 前缀回退重算 12 次。C2 将回退降至 0，少算 12288 tokens，重放 p95 比 C1 低 50.01%。6 个种子均改善，配对 p95 差为 −141.60 至 −132.73 ms。B 与 C2 均恢复了全部 48 条请求。[原始请求](../results/20261118_queue_c_evaluation/requests.jsonl) · [逐次决策](../results/20261118_queue_c_evaluation/decisions.jsonl) · [配对统计](../results/20261118_queue_c_report/summary.json)
- **目标请求更早出首 token，也会让整波更慢。** 在 48 个共同的恢复/重算观测上，目标 TTFT 预测的平均绝对误差从 C2 的 55.39 降至 C3 的 5.88 ms。24 波中有 6 波改选重算，均发生在第 4 条冷请求长 1600 tokens 时。恢复能让四条同一步执行，重算则把冷请求推到下一步。六对波中，C3 相对 B 的目标 TTFT 中位差为 −61.26 ms，冷请求为 +154.47 ms，波内 p95 为 +122.18 ms。全部 96 条重放的 p95 比 B 高 35.26%，6 个种子均更高。该负载采用 B 作为性能参照。[正式逐请求和逐步记录](../results/20261118_step_c_evaluation/measurements) · [两条路径逐波对照](../results/20261118_step_c_report/cases.jsonl) · [统计与补测口径](STAGE2_STEP_COST.md)

## 复核与使用

本次从归档逐请求日志离线复算。冻结对照、原版独立对照、全命中混合组和含冷请求组共核对 1854 条正式测量记录的首 token 时间戳，四个汇总命令生成的 7 个数据文件均与原归档逐字节一致。各负载单独统计。原版独立对照使用已核验输入一致的独立进程 B 补测，512/8192-token 及混合组的完整结果见原报告。[验收核对记录](../results/20261119_stage2_acceptance/verification.json) · [原版独立对照](STAGE2_RESTORE_POLICY.md)

在仓库根目录使用已有 `.venv` 执行以下命令，即可从原始记录重建汇总。临时目录可以保留用于逐项比较。

```bash
review_dir=$(mktemp -d)
.venv/bin/python -m benchmarks.compare_cpu_trace \
  --run results/20261118_stage2_b_pressure36 --output "$review_dir/ab.json"
.venv/bin/python -m benchmarks.compare_restore_policy \
  --calibration results/20261118_stage2_c_calibration --frozen-c results/20261118_stage2_c_pressure36 \
  --holdout results/20261118_stage2_c_holdout --holdout-b results/20261118_stage2_c_holdout_b_isolated \
  --output "$review_dir/c1"
.venv/bin/python -m benchmarks.queue_report \
  --calibration results/20261118_queue_c_calibration --evaluation results/20261118_queue_c_evaluation \
  --output "$review_dir/c2"
.venv/bin/python -m benchmarks.step_report \
  --calibration results/20261118_step_c_calibration --evaluation results/20261118_step_c_evaluation \
  --output "$review_dir/c3"
```

运行基于 `feat/pegaflow` 的 HEAD `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 及各实验目录内的未提交源码快照。B 的冻结性能数据采自尾块刷新修订前，384 次写入全部为新前缀，触发此次修订的写入为 0。完整 GPU 实验的命令和参数随各自快照保存，重测时将 `--output` 指向新目录。[B 原始命令](../results/20261118_stage2_b_pressure36/command.txt) · [B 写入审计](../results/20261118_stage2_b_pressure36/trace_comparison.json) · [C1 原始命令](../results/20261118_stage2_c_pressure36/command.txt) · [C2 原始命令](../results/20261118_queue_c_evaluation/command.txt) · [C3 原始命令](../results/20261118_step_c_evaluation/command.txt) · [C3 运行源码](../results/20261118_step_c_evaluation/source)

对外引用时保留比较对象、GPU 块数、并发、种子数、每策略样本量，并注明是重放段或整段。上述结论适用于这张卡和受控到达的负载。生产流量、其他硬件和异步搬运下的收益未测。
