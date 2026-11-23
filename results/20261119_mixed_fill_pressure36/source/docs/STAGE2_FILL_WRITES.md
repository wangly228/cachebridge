# 阶段 2 空缓存填充的 KV 写入优化

**冻结负载上，空缓存填充和整段 p95 都低于上一版串行写入。** 本轮同进程对照中，双缓冲使 24 条填充请求的引擎内 p95 TTFT 从 1284.73 降至 1258.65 ms（−2.03%），96 条完整负载从 1241.12 降至 1228.72 ms（−1.00%）。相对本轮原生 GPU 的无复用退化为 +1.43%，淘汰重放改善为 81.87%，满足原先的 5% 和 20% 目标。整段仍比原生 GPU 高 6.49%。[正式判定](../results/20261119_fill_batch_pressure36/status.json) · [分组汇总](../results/20261119_fill_batch_pressure36/summary.json)

## 改动

在已有的 `cpu_prefix_cache_blocks=64, cpu_prefix_admission="reuse"` 上设置 `cpu_prefix_batch_writes=True`。同一次 prefill 中，若两个以上完整 prompt 块都满足首次写入条件、键互异且有足够空闲 CPU slot，块管理器预留目的 slot 并交给 Runner 的传输对象。GPU 打包仍用一个 36 MiB 暂存块。在同一 CUDA 流上，将第一个块复制到锁页缓冲区，随后让下一块的 D2H 与上一块到最终 CPU slot 的主存复制重叠。两个锁页缓冲区轮换，总计 72 MiB。全部目的 slot 写完后才发布缓存映射，首 token 计时包含整组操作。[块管理与失败重试](../nanovllm/engine/cpu_prefix_cache.py) · [缓冲区与 CUDA event](../nanovllm/engine/kv_transfer.py)

遇到容量水位、单块、已有前缀刷新或混合准入时，按原顺序逐块处理。H2D 恢复继续沿用上一版。组内写入失败时先等在途传输结束，再归还所有预留 slot。可恢复的内存错误按单块重试。成本表会核对这项配置。默认配置保留单缓冲路径。[配置](../nanovllm/config.py) · [调度入口](../nanovllm/engine/scheduler.py) · [成本表匹配](../nanovllm/engine/restore_policy.py)

## 传输测量

RTX 3090，使用与 Qwen3-4B BF16 引擎相同布局的合成 KV，GPU 36 块、CPU 64 块，每组完整保存 8 个 36 MiB 块。每种路径计 12 组，交替顺序，预热、逐字节校验和 profiler 均在正式计时外。终点是数据已落入可分页 CPU slot。

| 同次对照的写入路径 | p50（ms） | p95（ms） |
| --- | ---: | ---: |
| 首轮，串行锁页暂存和 NumPy 主存复制 | 30.73 | 31.48 |
| 首轮，直接写入可分页张量 | 29.66 | 35.02 |
| 首轮，串行锁页暂存和 PyTorch 主存复制 | 38.03 | 39.65 |
| 第二轮，串行锁页暂存和 NumPy 主存复制 | 31.89 | 35.76 |
| 第二轮，双缓冲分组写入 | 25.98 | 31.51 |

直接写入可分页张量改善了 p50，但 p95 上升。双缓冲在同次对照中将 p50 降低 18.55%，p95 降低 11.90%。两轮分别对含预热的 312、208 次完整块写入做了逐字节校验。[首轮原始时间](../results/20261119_fill_copy_probe/batches.jsonl) · [双缓冲对照原始时间](../results/20261119_fill_copy_pipeline/batches.jsonl)

另取一组 8 块录制时间线。串行路径中，第 i 块的主存写入与第 i+1 块 D2H 重叠 0/7 对，双缓冲为 7/7 对，交集累计 20.22 ms。这是单组 profiler 观测，正式延迟用上表及逐请求日志。首次解析时 CPU 与 GPU 同名标注产生了重复记录，保留原始日志，并从原 trace 选取 CPU 外层标注、按相邻块重新离线汇总。[原时间线](../results/20261119_fill_copy_pipeline/copy_trace.json) · [各相邻块重叠时间](../results/20261119_fill_copy_pipeline/overlap.json) · [原控制台日志](../results/20261119_fill_copy_pipeline/console.log)

## 引擎内逐请求对照

2026-11-19，单卡 RTX 3090 24 GiB，Qwen3-4B BF16，revision `1cfa9a7208912126459214e8b04321603b3df60c`。冻结输入为每请求 2048-token 前缀加 128-token 尾部，greedy 输出 128 tokens，并发 4，prefill 步预算 2048 tokens，decode 启用 CUDA Graph。GPU 实际 36 块，两种 CPU 策略各 64 个 slot（2.25 GiB）。种子 101–103 各 2 个空缓存周期，三个策略在同一进程中轮换，覆盖全部 6 种次序。预先触碰主存页面，各周期开头清空两级索引，加载、预热和写盘在计时外。共享进程预分配两个锁页块，A 两个都闲置，串行策略只使用一个。正式对照关闭 profiler。[冻结输入](../results/20261118_pressure36/trace.json) · [实验设计](../results/20261119_fill_batch_pressure36/design.json) · [引擎配置](../results/20261119_fill_batch_pressure36/engine.json) · [环境与快照](../results/20261119_fill_batch_pressure36/environment.json)

下表单位为 ms，每条请求从入队计至首个真实输出 token。串行列使用上一项的 `reuse` 策略。两种 CPU 策略的每条请求输入、GPU/CPU 命中、保存与恢复块数、实际计算 tokens、输出 token IDs 全部相同。

| 场景 | 每策略请求数 | 原生 GPU A | 串行写入 | 双缓冲写入 | 双缓冲相对串行 | 双缓冲相对 A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 空缓存填充 | 24 | 1155.42 | 1284.73 | 1258.65 | −2.03% | +8.94% |
| GPU 热命中 | 24 | 78.60 | 79.88 | 79.63 | −0.31% | +1.31% |
| 无复用干扰 | 24 | 1163.57 | 1170.93 | 1180.16 | +0.79% | +1.43% |
| GPU 淘汰后重放 | 24 | 1151.16 | 208.32 | 208.73 | +0.20% | −81.87% |
| 四段合计 | 96 | 1153.83 | 1241.12 | 1228.72 | −1.00% | +6.49% |

- **搬运成本。** 两种 CPU 策略的填充段均写 192 块（6.75 GiB）。双缓冲实际执行 24 组，每组 8 块。累计写入时间从 752.94 降至 611.44 ms（−18.79%）。无复用段均逐块写入 24 块，淘汰重放均恢复 192 块。6 波填充的墙钟串行为 19.481 s、双缓冲为 19.099 s。6 个完整周期的墙钟 A / 串行 / 双缓冲为 67.789 / 63.635 / 63.084 s，整批输出吞吐为 181.27 / 193.10 / 194.79 tokens/s，包含 prefill 与 decode。[逐请求原始数据](../results/20261119_fill_batch_pressure36/requests.jsonl) · [周期墙钟](../results/20261119_fill_batch_pressure36/cycles.jsonl) · [逐波内存与耗时](../results/20261119_fill_batch_pressure36/waves.jsonl)
- **先淘汰再首次重放。** 每个种子的第二个周期均包含这一顺序。两个 CPU 策略在这 3 个周期的各 12 条重放中均恢复 96 块，每条恢复 2048 个前缀 tokens，只计算尾部 128 个。该组 p95 为 A 1158.17 ms、串行 208.33 ms、双缓冲 208.74 ms。[重放子组与分种子汇总](../results/20261119_fill_batch_pressure36/summary.json)
- **样本波动。** 每种策略共 6 个周期，每个种子每段 8 条相关请求。单独按种子算 p95，填充相对串行下降 1.73% 至 2.85%，整段下降 0.02% 至 2.87%。按相同种子和周期配对，6/6 个周期的填充 p95 更低，整段有 4/6 个更低，另外两个分别高 0.83、13.40 ms。这些是观察范围。整段按全部 96 条合并取 p95，最慢的 6 条仍全部来自填充。[逐周期离线核验](../results/20261119_fill_batch_pressure36/verification.json)

## 正确性与限制

在 GPU 3 块、CPU 4 块的 Qwen3-4B BF16 真机探针中，初始的两个完整块在同一组写入，随后两次淘汰恢复的 4 个 KV 块，以及对应的 8 行同形状完整 logits 均逐字节一致。正式对照的 288 条请求全部通过冻结同形状参照的 128-token greedy 输出核对，抢占与搬运失败均为 0。38 项测试通过，覆盖 slot 预留、半组失败后逐块重试、对齐尾块刷新，以及真实 CUDA 流上主存复制失败后重新使用两个缓冲区。新汇总和上一轮历史汇总均可从各自原始日志逐字节重建。[真机 KV](../results/20261119_fill_batch_model_probe/kv.jsonl) · [完整 logits 校验](../results/20261119_fill_batch_model_probe/logits.jsonl) · [测试日志](../results/20261119_fill_batch_tests.txt) · [冻结请求核验](../results/20261119_fill_batch_pressure36/verification.json)

这次收益来自同一 prefill 步中多个新块的写入，单块仍走串行路径。本轮使用 B 的命中即恢复。新模式下 C 的成本校准和收益未测，其他前缀长度、混合到达及生产流量的引擎端收益也未测。后续可用独立种子的 1024 / 2048-token 混合负载，保持并发 4 和 36 / 64 块容量，检查 4 块与 8 块写入并存时填充、整段 p95 是否仍低于串行，并复核无复用与淘汰重放的两项界限。

## 复跑

从仓库根目录使用既有模型和环境。GPU 命令需要新的输出目录，离线命令可直接读取本轮归档。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.fill_copy_probe --output results/fill_copy_repeat
.venv/bin/python -m benchmarks.fill_copy_probe --pipeline --output results/fill_pipeline_repeat
.venv/bin/python -m benchmarks.cpu_cache_probe --case fill-batch --output results/fill_model_repeat
.venv/bin/python -m benchmarks.write_admission --batched-fill --output results/fill_pressure_repeat
.venv/bin/python -m benchmarks.write_admission --summarize results/20261119_fill_batch_pressure36
.venv/bin/python results/20261119_fill_batch_pressure36/verify.py
```
