# 阶段 2 · 同步 CPU 前缀缓存 B

**策略 B 已接入 nano-vLLM。** 在 RTX 3090 上，GPU 淘汰后重放的 24 条请求全部从 CPU 恢复，首 token 延迟 p95 从 1239.99 ms 降至 248.64 ms，下降 79.95%。把填充、热命中和无复用干扰一并计入，96 条请求的整体 p95 从 1240.37 ms 升至 1404.63 ms，上升 13.24%。[分场景实测](../results/20261118_stage2_b_pressure36/summary.json) · [整条 trace](../results/20261118_stage2_b_pressure36/trace_comparison.json)

## 同一批请求的 A/B 实测

A 复用阶段 1 冻结的原始计时。B 直接读取 A 的输入 token IDs、请求顺序、预热输入和引擎配置，并核对归档 SHA256、模型文件、依赖版本及 GPU。两次测量都使用 Qwen3-4B BF16、TP=1、RTX 3090、GPU KV 池 36 块、每块 256 tokens、C=4、CUDA Graph decode、greedy、固定输出 128 tokens。每条 prompt 含 2048-token 前缀和 128-token 尾部。每个 seed 跑 2 个周期，每个周期开始时清空缓存索引，计入填充和干扰成本。模型加载、编译及预热在正式计时前完成，正式运行关闭 profiler。A、B 分时测量。[引擎配置与输入校验](../results/20261118_stage2_b_pressure36/engine.json) · [冻结 trace](../results/20261118_pressure36/trace.json)

| 场景 | 请求数 | A p95 TTFT (ms) | B p95 TTFT (ms) | B 相对 A | B 各 seed 的 p95 范围 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冷填充 | 24 | 1240.65 | 1391.97 | +12.20% | 1378.74–1397.98 |
| GPU 热命中 | 24 | 85.17 | 86.66 | +1.75% | 85.72–86.67 |
| 无复用干扰 | 24 | 1237.89 | 1438.86 | +16.24% | 1417.34–1560.18 |
| GPU 淘汰后重放 | 24 | 1239.99 | 248.64 | −79.95% | 214.29–248.66 |
| 四段合计 | 96 | 1240.37 | 1404.63 | +13.24% | 1390.47–1402.17 |

TTFT 从各请求 enqueue 计到首个真实输出 token。合计行用全部逐请求时间重算。每个场景的分 seed p95 各含 8 条请求，同波请求相关，表中区间是 3 个 seed 的观测范围。6 个周期合计墙钟由 71.775 s 降到 69.829 s（−2.71%），整段输出吞吐由 171.20 增至 175.97 tokens/s。该吞吐包含 prefill、decode、同步搬运和等待。[逐请求原始记录](../results/20261118_stage2_b_pressure36/requests.jsonl) · [逐周期墙钟](../results/20261118_stage2_b_pressure36/cycles.json)

- **恢复收益。** 在淘汰段，每条请求 GPU 命中 0 tokens，CPU 恢复 8 块共 2048 tokens，实际只计算剩下的 128 tokens。A 在同一段每条重算 2176 tokens。B 共恢复 192 块、6.75 GiB，同步 H2D 含主存复制和 GPU 写回共 0.816 s。[逐场景计数](../results/20261118_stage2_b_pressure36/summary.json)
- **保存代价。** 冷填充和无复用干扰各写入 192 块，两段合计 D2H 13.5 GiB，同步保存累计 1.906 s。无复用干扰 p95 上升 16.24%，超出设计里的 5% 退化目标。正式样本中抢占、保存失败和恢复失败均为 0。[逐场景计数](../results/20261118_stage2_b_pressure36/summary.json)
- **容量。** 每个周期包含 64 个不同的完整前缀块。本次选 64 个 CPU slot，pageable 数据区 2.25 GiB，另用 36 MiB pinned 主存和 36 MiB GPU 暂存。正式波结束时采样的最大进程 RSS 为 A 1.57 GiB、B 3.95 GiB，波内 GPU 分配峰值为 A 9.01 GiB、B 9.04 GiB。此 trace 的 CPU 淘汰次数为 0，容量满后的 LRU 淘汰在下述 4-slot 真机用例中单独验证。[内存与块数](../results/20261118_stage2_b_pressure36/trace_comparison.json)

## 引擎改动

通过 `LLM(model, cpu_prefix_cache_blocks=64)` 启用 B。默认值为 0，走原生 GPU 前缀缓存。CPU 层按完整块计容量。权重分片标识、模型配置（含 RoPE）、dtype、block size 和 TP 参与 namespace，每个引擎独占自己的 store。[配置](../nanovllm/config.py) · [块管理](../nanovllm/engine/cpu_prefix_cache.py)

完成计算的整块原始 prompt KV 在 GPU 上有有效引用时同步写入 CPU。分配时逐块验证前缀链和 token IDs，依次查 GPU 和 CPU，遇到缺口就结束复用，保留上游末尾 query 块。先占住全部 GPU 命中块和恢复目标，搬运完成后再发布可复用映射和 token 计数。CPU slot 按实际使用更新 LRU。可恢复的复制内存错误会撤销未完成映射，并按剩余前缀重新计算调度预算。[块管理](../nanovllm/engine/cpu_prefix_cache.py) · [调度器](../nanovllm/engine/scheduler.py)

Runner 按真实 KV 布局打包到一个 GPU 暂存块，经过一个 pinned 块与 pageable slot 同步交换数据。GPU 安全块数预算预留了这个暂存块。CPU 只保留历史副本，attention 始终读取 GPU KV。[搬运实现](../nanovllm/engine/kv_transfer.py) · [Runner](../nanovllm/engine/model_runner.py)

## 正确性与代码状态

- **真实 KV 和 logits。** 在 GPU 3 块、CPU 4 块、C=1 的 9 波用例中，目标前缀被 GPU 淘汰两次。两次恢复共核对 4 个 36 MiB 块，GPU 参考、CPU 副本、GPU 恢复值逐字节相同。相同输入和位置下，首 token 及后续 3 个位置，两次共 8 行完整 BF16 logits（每行 151936 项）逐字节相同。CPU LRU 共淘汰 6 次，最后一次重放按预期冷算。[KV 校验](../results/20261118_stage2_b_probe/kv.jsonl) · [logits 校验](../results/20261118_stage2_b_probe/logits.jsonl) · [运行判定](../results/20261118_stage2_b_probe/status.json)
- **整块尾部重算。** 长度恰为 512 时，GPU 命中前 256 tokens 后，末块重新计算。同一输入的旧、新末块有 15,128,983 个字节不同。缓存现会刷新该 slot。把 prompt 延长到 513 tokens 并制造 GPU 淘汰后，恢复块与最新 GPU 版本相差 0 字节，同位置完整 logits 也逐字节相同。压力实测归档于这项修订前。离线逐请求核对表明，正式 trace 的 384 次写入全是新前缀，触发此次修订的写入数为 0。[真机尾块校验](../results/20261118_stage2_b_full_tail/status.json) · [源块与副本 SHA256](../results/20261118_stage2_b_full_tail/kv.jsonl) · [正式 trace 写入核对](../results/20261118_stage2_b_pressure36/trace_comparison.json)
- **完整输出。** 正式 B 的 96/96 条请求与 A 在相同计算形状下的 128 个 greedy token 完全一致，其中 B 的淘汰后恢复对照 A 的 GPU 热命中。若直接对照 A 同一淘汰段的冷重算输出，则为 14/24 一致。另用最终代码默认关闭 CPU 层，冻结 A 的 8 条填充及热命中请求的完整输出和计算计数全部复现。[同形状判定](../results/20261118_stage2_b_pressure36/status.json) · [同场景对照](../results/20261118_stage2_b_pressure36/trace_comparison.json) · [默认路径](../results/20261118_stage2_default_off/status.json)
- **元数据与故障。** 原有 13 项测试通过。最终版本的 7 项缓存测试覆盖 255/256/257 边界、块对齐、跨层连续与部分命中、共享引用、CPU 满容量、重复恢复、抢占、生成块排除、hash 冲突以及注入的复制失败。[回归记录](../results/20261118_stage2_b_tests.txt) · [最终缓存测试](../results/20261118_stage2_b_final_cache_tests.txt)

## 复跑与下一步

在仓库根目录使用现有虚拟环境，GPU 实验的输出目录使用新路径。最后一条可直接从已有原始记录重算整条 trace 的 p95 和重复写入数。

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/pegaflow-inductor
export TRITON_CACHE_DIR=/tmp/pegaflow-triton
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.cpu_cache_probe --output results/new_cpu_probe
.venv/bin/python -m benchmarks.cpu_cache_probe --case aligned-tail --output results/new_cpu_tail
.venv/bin/python -m benchmarks.cpu_cache_probe --case default-off --output results/new_gpu_only
.venv/bin/python -m benchmarks.cpu_cache_pressure --output results/new_cpu_pressure
.venv/bin/python -m benchmarks.compare_cpu_trace --run results/20261118_stage2_b_pressure36 --output /tmp/pegaflow_ab_trace.json
```

B 实验结束时确定的下一步为策略 C。此后已完成 512、2048、8192-token 前缀的独立成本校准、冻结 trace 的 A/B/C 对照和独立混合负载验证。逐请求预测、实际恢复及重算开销见 [策略 C 实测报告](STAGE2_RESTORE_POLICY.md)。
