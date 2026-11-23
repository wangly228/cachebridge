# 阶段 2 优化：减少无复用前缀的同步写入

**冻结负载上的两个目标均已达到。** 2026-11-19 在同一台 RTX 3090 上轮换测量，相对本轮原生 GPU 基线 A，新策略在 24 条无复用干扰请求上的引擎内 p95 TTFT 上升 **1.11%**（上限 5%），在 24 条 GPU 淘汰重放上下降 **82.12%**（目标至少 20%）。3 个种子分别都达标。96 条请求的整段 p95 仍上升 **8.49%**，最慢的 6 条全部来自空缓存填充。[判定](../results/20261119_reuse_admission_pressure36/status.json) · [逐请求分组汇总](../results/20261119_reuse_admission_pressure36/summary.json)

## 如何决定写入

在正常的 `LLM(..., cpu_prefix_cache_blocks=64, cpu_prefix_admission="reuse")` 入口启用新策略。它沿用 B 的同步传输和 CPU 命中即恢复规则。`cpu_prefix_admission="always"` 是原来启用 CPU 缓存时的默认行为。CPU 缓存仍可通过 0 个 slot 关闭。[配置](../nanovllm/config.py) · [块管理器](../nanovllm/engine/cpu_prefix_cache.py)

1. 从空缓存开始，对完整、已计算的 prompt 块直接保存，直到 CPU 驻留块数达到 GPU 块数与 CPU slot 数中的较小值。本次为 36 块。这为第一次复用前就发生 GPU 淘汰的前缀保留了恢复机会。
2. 达到该水位后，新块第一次出现只记前缀哈希和请求 ID。最近使用记录按 LRU 有界淘汰，上限等于 CPU slot 数，本次为 64 条。另一个请求在记录有效时真正使用该块才触发 D2H。GPU 命中时从被引用保护的页复制，GPU 已淘汰时从本次重算完成的页复制。调度试探和同一请求抢占后的重试都不会充当第二个请求。
3. CPU 中已有的完整 prompt 块一旦重算，立即刷新到最新 GPU 版本。一次分配先完成所有 H2D，再允许新写入占用 LRU 淘汰的 slot。校验前缀链、token IDs、保留末尾 query 的语义与 B 相同。C 的成本表校验包含写入策略，本次性能实验使用 B 的恢复规则。[调度入口](../nanovllm/engine/scheduler.py) · [成本表校验](../nanovllm/engine/restore_policy.py)

历史冻结负载中，原 B 在无复用干扰段写入 192 个块（6.75 GiB），这些前缀在各自周期内后续出现 0 次。每个种子有两个周期，其中一个先 GPU 热命中再淘汰，另一个先淘汰再首次重放。新策略只使用当前缓存占用和已经发生的访问。[冻结 token IDs 与波次](../results/20261118_pressure36/trace.json) · [历史 B 汇总](../results/20261118_stage2_b_pressure36/summary.json)

## 同进程对照

Qwen3-4B BF16，revision `1cfa9a7208912126459214e8b04321603b3df60c`，单卡 RTX 3090 24 GiB。每块为 256 tokens / 36 MiB。GPU 实际 36 块，两种 CPU 策略均为 64 个 slot（2.25 GiB），各有 36 MiB 锁页主存和 GPU 暂存。并发 4，每请求 2048-token 前缀加 128-token 尾部，greedy 输出 128 tokens，CUDA Graph 开启。种子 101–103 各 2 个周期，每个周期以相同到达顺序和输入 IDs 跑三个策略。6 个周期覆盖全部 6 种策略顺序。各策略每周期开头清空 GPU/CPU 索引和访问历史，预热和写盘在计时外。A 使用原生块管理器，共用进程中为 CPU 策略分配的缓冲区闲置。进程预先触碰主存页面，加载与首次分配不计入首 token 等待时间。[实验设计与输入校验和](../results/20261119_reuse_admission_pressure36/design.json) · [实际引擎配置](../results/20261119_reuse_admission_pressure36/engine.json) · [环境与源码快照清单](../results/20261119_reuse_admission_pressure36/environment.json)

以下为从入队到首个真实生成 token 的 p95，单位 ms。原 B 对每个完整 prompt 块同步写入。新策略使用相同的 H2D 恢复规则。整段 p95 取全部 96 条请求的分位数。

| 场景 | 每策略请求数 | 本轮 A | 本轮原 B | 新策略 | 新策略相对本轮 A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 空缓存填充 | 24 | 1187.10 | 1299.65 | 1297.69 | +9.32% |
| GPU 热命中 | 24 | 81.15 | 81.04 | 80.97 | −0.23% |
| 无复用干扰 | 24 | 1172.79 | 1285.11 | 1185.83 | +1.11% |
| GPU 淘汰后重放 | 24 | 1178.03 | 205.57 | 210.59 | −82.12% |
| 四段合计 | 96 | 1177.24 | 1286.71 | 1277.24 | +8.49% |

- **少写了多少。** 无复用段，原 B 写 192 块，新策略写 24 块（0.844 GiB），减少 87.5%。该段同步 D2H 累计耗时由 728.76 降至 91.15 ms。6 个周期合计的 D2H 从 384 减至 216 块。两种 CPU 策略在 24 条淘汰重放上都实际恢复了 192 块，每条只算 128 个尾部 tokens。6 波干扰结束时，原 B 每次驻留 64 块，新策略每次为 36 块。[逐请求传输与计算](../results/20261119_reuse_admission_pressure36/requests.jsonl) · [逐波缓存与内存](../results/20261119_reuse_admission_pressure36/waves.jsonl)
- **首轮填充仍要付费。** 两种 CPU 策略在填充段均写 192 块。新策略累计花 745.80 ms 做这些同步复制，6 波填充的墙钟为 19.602 s，A 为 18.888 s。按 TTFT 从低到高排序，最后 6 条均为填充请求，决定合并 p95 的第 91、92 条也在其中。6 个完整周期的墙钟 A / 原 B / 新策略为 69.092 / 64.388 / 64.117 s，整段输出吞吐为 177.85 / 190.84 / 191.65 tokens/s，包含 prefill 和 decode。[周期墙钟](../results/20261119_reuse_admission_pressure36/cycles.jsonl) · [逐请求排序核验](../results/20261119_reuse_admission_pressure36/verification.json)
- **样本与不确定性。** 每个种子每段 8 条相关请求。按种子分别计算 p95 后，新策略相对同种子 A 的无复用变化范围为 −0.76% 至 +1.66%，淘汰重放为 −82.66% 至 −82.08%。这些是 3 个种子的观察范围。改以之前冻结的 A 绝对值为分母，新策略的无复用、淘汰重放、整段分别为 −4.20%、−83.02%、+2.97%；本轮只用同进程轮换的数据估计改动的影响。原 B 在本轮的无复用退化为 +9.58%，历史记录为 +16.24%。[分种子和双参照汇总](../results/20261119_reuse_admission_pressure36/summary.json) · [历史验收](STAGE2_ACCEPTANCE.md)

## 正确性与适用边界

在 RTX 3090、Qwen3-4B BF16、GPU 3 块、CPU 4 块的 14 波检查中，候选前缀从 GPU 提升到 CPU 后，两次淘汰恢复的 4 个 KV 块、相同输入及位置的 8 行完整 logits 均逐字节一致。同一个检查还制造了相反的访问顺序：首次保存额度已满时，一个只见过一次的前缀在再次访问前被 GPU 淘汰。第二次请求冷算 513 tokens，写入 2 块；再次淘汰后，下次请求从 CPU 恢复 512 tokens，只计算尾部 1 token。候选记录被挤出后也需要重新累计访问次数。[逐请求记录](../results/20261119_reuse_admission_probe/requests.jsonl) · [KV 校验](../results/20261119_reuse_admission_probe/kv.jsonl) · [logits 校验](../results/20261119_reuse_admission_probe/logits.jsonl) · [单元测试](../tests/test_cpu_prefix_cache.py)

正式对照共 288 条请求，全部通过同形状参照的 128 个 greedy 输出 token 核对。CPU 恢复与冻结 A 的 GPU 热命中使用相同输入和执行形状。所有策略的抢占与复制失败均为 0，35 项单元测试通过。同期 GPU/主机峰值、逐步调度、TPOT、ITL 在原始记录和汇总中可查。生产流量及新写入策略下的 C 成本校准未测。[正式判定](../results/20261119_reuse_admission_pressure36/status.json) · [单元测试日志](../results/20261119_reuse_admission_tests.txt) · [逐步记录](../results/20261119_reuse_admission_pressure36/steps.jsonl)

下一项可验证的优化是降低首次填充的同步写入成本。对照同一负载的 24 条空缓存填充和 96 条整段请求，同时检查无复用退化仍在 5% 内、淘汰重放改善仍至少为 20%，单独核对先淘汰再首次重放的 3 个周期。

## 复跑

在仓库根目录使用已有环境。GPU 命令各需一个新的输出目录，离线汇总可以直接读取归档。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.cpu_cache_probe --case reuse-admission --output results/reuse_probe_repeat
.venv/bin/python -m benchmarks.write_admission --output results/reuse_pressure_repeat
.venv/bin/python -m benchmarks.write_admission --summarize results/20261119_reuse_admission_pressure36
```
