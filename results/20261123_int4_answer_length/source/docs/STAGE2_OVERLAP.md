# 阶段 2 首次 KV 写入与后续计算重叠

**保留显式启用的后台写入。** 在单张 RTX 3090、Qwen3-4B BF16、6 个配对种子上，混合负载含冷填充的整段 p95 TTFT 从同步 32 块的 938.66 ms 降至 908.70 ms，下降 3.19%，配对种子 95% 区间为 −4.57% 至 −1.63%，每臂 240 条请求。同轮原生 A 为 895.28 ms，重叠写入仍比 A 高 1.50%（+0.58% 至 +2.12%）。全长负载每臂 192 条，重叠比同步低 5.12%，比 A 高 1.84%（+0.58% 至 +2.96%）。两种负载的淘汰重放 p95 分别比 A 低 80.55% 和 82.29%。[正式汇总及原始文件摘要](../results/20261123_overlap_formal/summary.json)

## 确认阻塞位置

历史 32 块与 36 块在混合冷填充的 p95 相差 16.69 ms，两档每波写入相同的 24 块。2026-11-23 复测种子 1803 周期 0、1804 周期 1 的冻结输入，先交替重复相同的 32 块配置，再交替比较额度。关闭分段诊断时，每组每臂 8 波、32 条请求，相同 32 的合并 p95 差为 5.34 ms，32 与 36 的差为 0.99 ms。32 比原生 A 多 73.79 ms，8 对波全部更慢。另开诊断轮分别计时输入准备、前向及 logits、采样、token ID 回传和写入。CPU 组 20 波共有 80 次写入，全部发生在同波队尾首 token 之前，耗时中位数 19.10 ms，紧随其后的 prefill 前向及 logits 耗时中位数 163.18 ms。基于该窗口，本轮只深入验证后台写入一个候选。[历史冷填充分解](STAGE2_FIRST_USE_BUDGET.md#混合冷填充耗时分解) · [双波诊断与复跑](STAGE2_COLD_FILL_PROBE.md) · [逐次时间线](../results/20261123_cold_fill_probe/timeline.jsonl)

## 引擎改动

启用 `cpu_prefix_transfer_mode="overlap"` 且打开批量写入后，块管理器为同一请求的至少两个完整新 prompt 块预留空 CPU slot。搬运线程在独立 CUDA stream 上等待生产这些 KV 的事件，然后复用原有 GPU 打包、双锁页缓冲 D2H 和 CPU slot 复制逻辑。主线程可以在此期间执行后续推理。引擎最多保留一批在途写入，完成整批复制后才发布 CPU 映射。活跃请求持有源块，回收、抢占、可能复用缓冲区的命中分配、下一批保存及关闭引擎均会在需要时等候。单块写入、满容量 LRU 和 H2D 恢复沿用同步路径。常规调用省略传输模式时仍使用 `sync`。[块管理器](../nanovllm/engine/cpu_prefix_cache.py) · [真实搬运](../nanovllm/engine/kv_transfer.py) · [配置项](../nanovllm/config.py)

按本轮配置启用重叠写入的最小示例如下，`MODEL_DIR` 指向已有本地模型目录。

```python
import os
from nanovllm import LLM

llm = LLM(
    os.environ["MODEL_DIR"],
    max_kvcache_blocks=36,
    cpu_prefix_cache_blocks=64,
    cpu_prefix_admission="reuse",
    cpu_prefix_first_use_blocks=32,
    cpu_prefix_batch_writes=True,
    cpu_prefix_transfer_mode="overlap",
)
```

47 项单元测试通过。补强提交失败后的资源回收后，7 项后台写入测试复核通过。覆盖复制期间的源块引用和 CPU slot 预留、部分失败重试、硬错误回收、满容量 LRU，以及真实 CUDA 侧流上的 BF16 D2H→H2D 逐位一致性。[首轮测试](../results/20261123_overlap_tests.txt) · [定向复核](../results/20261123_overlap_tests_recheck.txt) · [测试源码](../tests/test_overlap_writes.py)

## 同轮对照与首次筛查

三臂共用模型与张量池。A 使用原生 GPU 前缀缓存，GPU 未命中时重算。同步与重叠两臂采用相同的 32 块首次写入额度、按复用写入策略、CPU 容量和命中即恢复逻辑。每个对照臂开始前重建空 GPU/CPU 索引。计时从每条请求入队到引擎记录其首个真实输出 token。每波运行至所有输出完成，同步设备并检查在途写入已清空。整批吞吐还计入各周期全部搬运和收尾。模型加载、编译、CUDA Graph 捕获、主存预触碰、预热及日志写盘在正式推理计时外。[对照入口与检查](../benchmarks/transfer_comparison.py) · [引擎配置](../results/20261123_overlap_formal/engine.json) · [预设验收线](../results/20261123_overlap_formal/design.json)

独立筛查种子 1899 每臂 20 条混合、16 条全长请求，均使用先淘汰再重放的顺序。重叠相对同步的混合整段 p95 下降 3.18%，全长整段上升 3.86%。混合淘汰重放、全长无复用和全长淘汰重放分别上升 7.12%、5.95%、9.75%，三个子组每臂各 4 条。108 条实测包含 72 条 CPU 策略请求，逐条同形状原生 greedy 对照通过。主指标出现改善，据此进入独立多种子验收。[筛查结果](../results/20261123_overlap_screen/summary.json)

## 六种子正式验收

单卡 RTX 3090 24 GiB，Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`，权重与 KV 均为 BF16。每块 256 tokens、36 MiB，GPU 36 块、CPU 64 块。每波并发 4，混合负载两条 1024 和两条 2048 前缀，全长负载四条 2048 前缀，每条追加 128 个输入 tokens，以 greedy 生成 128 个输出。prefill 每步上限 2048 tokens，CUDA Graph 开启，profiler 关闭。每个种子既测“填充 → GPU 热重放 → 干扰 → 淘汰重放”，也测“填充 → 干扰 → 淘汰重放 → GPU 热重放”。混合用两波无复用请求造成淘汰，全长用一波。[模型、依赖与源码哈希](../results/20261123_overlap_formal/environment.json) · [完整输入与执行顺序](../results/20261123_overlap_formal/trace.json)

种子 1901–1906，每个“负载 × 访问顺序”下的三臂六种排列恰好各用一次。下表单位为 ms，各行从该范围的全部请求直接取 p95。“先淘汰再首次重放”是淘汰重放的子集。区间按完整种子成对重采样 4000 次，每次保留两个周期和全部 FIFO 位置，然后重算合并 p95 的相对差。6 个种子中，两个负载各自的整段 p95 均是重叠低于同步。区间仅反映这组输入、访问模式与设备上的种子差异。[逐请求首 token 时间戳](../results/20261123_overlap_formal/requests.jsonl) · [各段与逐种子结果](../results/20261123_overlap_formal/summary.json)

| 负载 | 范围 | 每臂请求 | 原生 A | 同步 32 | 重叠 32 | 重叠比同步 | 配对 95% 区间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 混合 | 整段，含冷填充 | 240 | 895.28 | 938.66 | 908.70 | −3.19% | −4.57% 至 −1.63% |
| 混合 | 冷填充 | 48 | 890.51 | 973.66 | 925.85 | −4.91% | −6.59% 至 −2.96% |
| 混合 | 填充后 | 192 | 896.46 | 907.26 | 901.87 | −0.59% | −1.95% 至 −0.13% |
| 混合 | GPU 热重放 | 48 | 77.57 | 77.92 | 78.76 | +1.08% | −1.15% 至 +2.34% |
| 混合 | 无复用 | 96 | 896.90 | 923.29 | 912.11 | −1.21% | −2.05% 至 +1.07% |
| 混合 | 淘汰重放 | 48 | 903.62 | 184.95 | 175.76 | −4.97% | −5.67% 至 −0.31% |
| 混合 | 先淘汰再首次重放 | 24 | 899.03 | 184.96 | 175.76 | −4.97% | −5.67% 至 +2.78% |
| 全长 | 整段，含冷填充 | 192 | 1181.03 | 1267.70 | 1202.76 | −5.12% | −6.15% 至 −3.98% |
| 全长 | 冷填充 | 48 | 1176.44 | 1282.63 | 1221.46 | −4.77% | −5.57% 至 −4.48% |
| 全长 | 填充后 | 144 | 1182.34 | 1166.63 | 1175.11 | +0.73% | −0.18% 至 +2.02% |
| 全长 | GPU 热重放 | 48 | 82.18 | 80.97 | 81.31 | +0.43% | −0.98% 至 +1.59% |
| 全长 | 无复用 | 48 | 1183.74 | 1177.37 | 1191.17 | +1.17% | −0.19% 至 +1.87% |
| 全长 | 淘汰重放 | 48 | 1190.39 | 210.64 | 210.85 | +0.10% | +0.10% 至 +2.16% |
| 全长 | 先淘汰再首次重放 | 24 | 1195.09 | 208.51 | 207.52 | −0.47% | −2.17% 至 +0.62% |

验收要求混合整段 p95 降低至少 3%，配对种子 95% 区间上界低于零，两负载无复用和淘汰重放 p95 相对同步的合并点估计退化均不超过 5%。这些条件全部达到。重叠把混合冷填充 p95 从 973.66 ms 降到 925.85 ms，全长从 1282.63 ms 降到 1221.46 ms。全长无复用仍退化 1.17%，淘汰重放退化 0.10%。两种负载的 GPU 热重放点估计也分别高 1.08% 和 0.43%。[正式验收判定](../results/20261123_overlap_formal/summary.json)

## 搬运、吞吐与占用

每个负载共 12 个冷启动周期。下表的 D2H 为写入 CPU 的块数，H2D 为从 CPU 恢复的块数。GPU / CPU 命中按初始命中请求计。整批输出吞吐为输出 tokens 总数除以周期墙钟总和，包含 prefill、decode、搬运和等待。[逐周期墙钟](../results/20261123_overlap_formal/cycles.jsonl) · [逐波吞吐与内存](../results/20261123_overlap_formal/waves.jsonl)

| 负载 | 方案 | 整段墙钟（s） | 整批输出（tokens/s） | GPU / CPU 命中请求 | 实际计算 prompt tokens | D2H / H2D 块 | 淘汰重放完整恢复 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 混合 | 原生 A | 163.48 | 187.91 | 48 / 0 | 325632 | 0 / 0 | 0 / 48 |
| 混合 | 同步 32 | 156.45 | 196.36 | 48 / 48 | 251904 | 384 / 288 | 48 / 48 |
| 混合 | 重叠 32 | 156.05 | 196.86 | 48 / 48 | 251904 | 384 / 288 | 48 / 48 |
| 全长 | 原生 A | 139.70 | 175.93 | 48 / 0 | 319488 | 0 / 0 | 0 / 48 |
| 全长 | 同步 32 | 128.89 | 190.67 | 48 / 48 | 221184 | 384 / 384 | 48 / 48 |
| 全长 | 重叠 32 | 128.87 | 190.71 | 48 / 48 | 221184 | 384 / 384 | 48 / 48 |

两个 CPU 臂在“先淘汰再首次重放”子组各自完整恢复混合 24/24 条、144/144 块，全长 24/24 条、192/192 块。1296 条正式请求的 TTFT 可从时间戳复算。全部 864 条 CPU 策略请求逐条核对同形状原生 GPU 的调度步、计算量和全部 128 个 greedy 输出。CPU 恢复的淘汰重放使用相同完整输入的原生 GPU 热重放作为参照。各臂均为零抢占，两 CPU 臂均为零保存失败、零恢复失败、零 CPU 淘汰。另有 108 条预热请求排除在正式统计外。[逐步入批与计算量](../results/20261123_overlap_formal/steps.jsonl) · [逐请求恢复计数与输出](../results/20261123_overlap_formal/requests.jsonl)

三臂记录的 GPU 张量分配峰值均为 9.04 GiB，进程 CPU RSS 高水位为 4.66 GiB。模型与搬运缓冲在同一进程中预分配，A 也保留这些张量。两个 CPU 臂均使用 2.25 GiB CPU slot、72 MiB 锁页暂存、36 MiB GPU 暂存。重叠模式另有一条工作线程和一条 CUDA stream。GPU 活跃块峰值混合 28、全长 36。此次性能测量覆盖无抢占的固定到达批次，多机、其他模型和持续在线到达的收益未测。[逐波内存](../results/20261123_overlap_formal/waves.jsonl) · [测量时搬运实现](../results/20261123_overlap_formal/source/nanovllm/engine/kv_transfer.py)

## 复跑

从公开仓库根目录启动已有 Python 环境。下面使用本机实测的解释器相对路径。两条 GPU 实验命令需要新的输出目录，最后一条读取已归档数据，离线重建三臂摘要。诊断双波实验的命令另见诊断报告。公开分支本轮测量基点为 `e69c824279c0da2b10f5078d0174b80f0fff03d5`，未提交的引擎与基准改动已随运行保存。[正式命令](../results/20261123_overlap_formal/command.txt) · [运行时源码](../results/20261123_overlap_formal/source/) · [工作区差异](../results/20261123_overlap_formal/git.diff) · [诊断复跑](STAGE2_COLD_FILL_PROBE.md#复跑)

```bash
py=../PegaFlow/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.transfer_comparison --stage screen --candidate overlap --output results/overlap_screen_repeat
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.transfer_comparison --stage formal --candidate overlap --output results/overlap_formal_repeat
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.transfer_comparison --summarize results/20261123_overlap_formal
```
