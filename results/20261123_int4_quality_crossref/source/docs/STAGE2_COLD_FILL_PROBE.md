# 混合冷填充的额度波动与写入代价

**结束 32 与 36 块的冷填充额度排查，候选优化转向同步 KV 写入。** 2026-11-23 在两波冻结输入上，关闭诊断时，32 块相对 36 块的合并 p95 高 0.99 ms。两份相同的 32 块配置相差 5.34 ms。32 块相对原生 GPU 基线 A 高 73.79 ms，8 个配对波全部更慢，每波同步写入平均占 77.51 ms。[逐组结果和配对波差值](../results/20261123_cold_fill_probe/summary.json)

## 两波复测

使用混合负载种子 1803 周期 0、1804 周期 1 的原始填充波，每波 4 条请求，1024 和 2048 两种前缀各两条，分别追加 128 个尾部输入 tokens，正式测量均生成 128 个 greedy tokens。每波重建空 GPU/CPU 缓存索引。RTX 3090 24 GB、Qwen3-4B BF16、单卡、CUDA Graph 开启、每步 prefill 预算 2048 tokens，GPU 实际 36 块、CPU 容量 64 块、块长 256 tokens。模型文件 SHA256、依赖版本和 CUDA runtime 与上轮记录匹配。A 与 CPU 组共用模型、GPU 池及已分配的搬运缓冲，A 使用原生 GPU 块管理器。[实际配置](../results/20261123_cold_fill_probe/engine.json) · [软硬件与模型](../results/20261123_cold_fill_probe/environment.json) · [固定输入](../results/20261119_first_use_budget36/trace.json)

先重复同一配置，再比较额度，最后比较原生 A。每种对照先关闭诊断重复 4 轮，再开启诊断重复 2 轮。每轮两波，每波轮换两个对照臂的执行先后。两波各自的 8-token 预热排除后，共计 72 波、288 条请求。下表全部采用未插入分段同步的端到端测量。每组每臂 8 波、32 条，合并 p95 取所有请求自入队至第一个真实输出 token 的时间。最后一列先在每个四请求波内取 p95，再按相同输入与轮次求差。[测量顺序](../results/20261123_cold_fill_probe/design.json) · [逐请求时间戳与输出](../results/20261123_cold_fill_probe/requests.jsonl)

| 配对 | 左臂合并 p95，ms | 右臂合并 p95，ms | 合并差，ms | 8 对逐波差范围，ms |
| --- | ---: | ---: | ---: | ---: |
| 32 与相同 32 | 999.87 | 994.53 | +5.34 | −10.12 至 +15.53 |
| 32 与 36 | 1010.64 | 1009.65 | +0.99 | −10.41 至 +9.48 |
| 32 与 A | 1005.65 | 931.87 | +73.79 | +66.13 至 +83.72 |

32 和 36 每波都写 24 个块，均未触及额度上限。两档合并差和逐波最大绝对差都小于重复 32 的对应波动。诊断打开时，32 相对 36 的合并差换为 −4.98 ms。历史额度差的具体成因未定位。本次只有两个冻结输入波，各轮观测的是这两波的重复运行波动。[逐波记录](../results/20261123_cold_fill_probe/waves.jsonl)

## 同步写入和后续计算

诊断轮仅在 prefill 内分段。输入准备含采样温度与输入张量准备、传输，前向含 logits 计算，采样含 greedy 选 token。在每段结束时同步设备，再单独计时 token ID 回到主机。写入记录覆盖整次同步 `save_many`，包括 GPU 打包、D2H、等待和最终 CPU slot 复制。下表是每波各 prefill 步的累计时间，再按波取平均；每个标签各 4 波、16 条请求。合并 p95 使用上表的关闭诊断轮。[分段时间线](../results/20261123_cold_fill_probe/timeline.jsonl) · [引擎调度步](../results/20261123_cold_fill_probe/steps.jsonl)

| 同轮比较 | 标签 | 输入，ms | 前向及 logits，ms | 采样，ms | ID 回传，ms | 同步写入，ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 相同 32 | 32a | 3.18 | 913.17 | 3.57 | 0.58 | 77.37 |
| 相同 32 | 32b | 3.58 | 915.29 | 3.63 | 0.64 | 78.82 |
| 额度 | 32 | 3.36 | 915.14 | 3.66 | 0.63 | 77.76 |
| 额度 | 36 | 3.40 | 917.46 | 3.58 | 0.65 | 77.12 |
| 原生对照 | 32 | 3.15 | 881.55 | 3.61 | 0.68 | 74.87 |
| 原生对照 | A | 2.59 | 878.56 | 3.74 | 0.68 | 0.00 |

两波都按 FIFO 顺序处理。诊断中的 CPU 组共 20 波，80 次同步写入全在队尾请求发布首 token 前完成。每次写入耗时 12.18 至 28.68 ms（中位 19.10 ms），之后紧接的下一次 prefill 前向及 logits 耗时 38.05 至 261.34 ms（中位 163.18 ms）。该时间线给出了尝试边写入边执行下一步计算的窗口。两者并行时的带宽竞争和整段收益需要以独立候选对照测量。[每批物理块、slot、后继前向时间](../results/20261123_cold_fill_probe/timeline.jsonl)

本次 288 条测量请求的入批形状和完整 greedy 输出在相同冻结输入的各臂一致。另从原生 A 归档逐条复核本次输入、实际 prefill/decode tokens、初始 GPU/CPU 命中、抢占、保存和恢复失败以及 36,864 个输出 token ID，全部一致。[归档原生对照](../results/20261119_first_use_budget36/requests.jsonl) · [本次请求](../results/20261123_cold_fill_probe/requests.jsonl)

## 复跑

在公开仓库根目录使用现有环境。第一次命令在新的输出目录运行全部 GPU 轮次；第二条从本次归档重建汇总。

```bash
PYTHONDONTWRITEBYTECODE=1 ../PegaFlow/.venv/bin/python -m benchmarks.cold_fill_probe --output results/cold_fill_probe_repeat
PYTHONDONTWRITEBYTECODE=1 ../PegaFlow/.venv/bin/python -m benchmarks.cold_fill_probe --summarize results/20261123_cold_fill_probe
```
