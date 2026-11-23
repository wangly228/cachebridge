# 阶段 1 本地核验与初测（2026-11-17）

**已完成指标、KV cap、边界测试和第一轮成本测量；阶段 1 尚未冻结。** 本轮保留了两次失败：并发 4 的一个冷/热缓存 greedy 分歧，以及 32 块一次性 pinned allocation OOM。没有实现 CPU 前缀缓存，也没有安装、升级依赖或下载模型。

2026-11-18 后续进展见 [固定路径核验报告](FIXED_PATH_REPORT.md)：前缀/真实 KV 搬运的精确检查通过，冷/热数值筛查仍有 6/144 个位置失败。以下保留 2026-11-17 的初测与当时验收状态，不改写历史结果。

## 本地核验

- 分支 `feat/pegaflow`，HEAD `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，remote 为 `GeeeekExplorer/nano-vllm`。开始时只有用户提供的 `AGENTS.md`、`docs/` 未跟踪，源码无修改。未切分支、提交或推送。
- 使用仓库 `.venv/bin/python`（3.12.15），editable `nano-vllm==0.2.0` 指向本仓库。Torch 2.10.0+cu128、Triton 3.6.0、FlashAttention 2.8.3+cu.12.8.torch.2.10、Transformers 4.57.6、NumPy 2.5.5、xxhash 4.1.0、einops 0.8.2、safetensors 0.8.0。
- WSL2 Ubuntu 24.04.5；可见主存约 31 GiB、swap 8 GiB。RTX 3090 24 GiB，compute capability 8.6，BF16 可用；驱动 610.57.01 / KMD 610.88。最初沙箱拒绝 GPU 访问，获准在沙箱外核验和实验后正常运行，非环境损坏。
- 本地模型 `~/models/huggingface/Qwen3-4B`。下载元数据 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`；本轮计算全部模型文件 SHA256，三个 safetensors shard 的 SHA256 均与下载元数据一致。权重文件 tensor dtype、加载后权重、实际 KV tensor 均为 BF16。
- 搜索仓库及 workspace 的脚本/日志，只有上游 `bench.py`，没有找到历史三轮吞吐初测的原始脚本和日志。因此历史 81–102 tok/s 仍仅作为历史记录，不回填新配置。

详见各 run 的 `environment.json`（含模型哈希、依赖、GPU、主存）、`git.diff`、`source/` 快照和 `command.txt`。由于没有提交，新文件也被复制进 source 快照，不能仅靠 HEAD 复原实验。

## 改动与口径

| 改动 | 行为与边界 |
| --- | --- |
| `enable_metrics=True` | 默认关闭；逐请求记录 enqueue、真实 token append、finish、TTFT/ITL/TPOT、实际 prefill/原 prompt 计算 tokens、decode tokens、首次 GPU 命中、再次分配命中和抢占。调用 `clear_metrics()` 释放累计记录，要求队列已空。 |
| 首 token 时间 | 分词之后、构造请求之前开始 enqueue 计时；`ModelRunner.run()` 已通过 `.tolist()` 将输出 ID 带回 host，随后在实际 append 处记录。Chunked Prefill 中间块不记 token。每个 token 没有新增全局 CUDA synchronize。 |
| step 计时 | `run_ms` 包含输入准备、模型、采样、输出 ID materialization；`step_ms` 还包含调度与 postprocess。它们不是纯 kernel 时间。 |
| `max_kvcache_blocks` | `min(安全预算块数, 指定上限)`；记录 safe/requested/actual。本轮 safe=288、cap=160、actual=160，每块 36 MiB，总 KV 5.625 GiB。 |
| admission | 空输入、超过 model length 或单请求连完整 KV 都放不下时显式拒绝，避免把调度失败算作延迟。末个输出尚未作为 query，无需其 KV slot。 |
| greedy | 新增明确的 `temperature=0` argmax 分支，除法使用安全温度；正温度保留采样路径。BF16 在不同计算形状下仍可能改变并列 logits 的 argmax。 |

原生 GPU cache、hash 链、ref_count、最后一块保守排除、调度顺序、Chunked Prefill 和 CUDA Graph 实现均沿用上游；未新增 CPU store、LRU、恢复策略或异步传输。

## 实验配置与结果

全部为本地单卡 BF16、TP=1、block size=256。引擎实验统一 eager、`max_model_len=9216`、`max_num_batched_tokens=2048`、`max_num_seqs=8`、`gpu_memory_utilization=0.8`、KV cap=160。实际配置见 `engine.json`。

输入为固定 seed 生成的随机合法 token IDs（100–9999），原始输入/输出均保存在 `requests.jsonl`。不是自然语言质量评测。每波按固定顺序连续 enqueue C 条请求，等本波全部结束再开始下一波；C 是到达波并发，不能等同每步模型 batch size。采样固定 greedy、ignore EOS。

加载、编译和形状预热在测量外；预热阶段记录为 `warmup`。每对实验先在队列为空时替换 BlockManager 元数据清冷缓存，cold 执行同时填充原生 GPU cache，随后 warm 重放相同输入。原始冷执行成本另存，不把 warm 称为含填充成本的整段收益。波与波之间的 JSON 写盘、输入构造、缓存重置不计入单波推理墙钟。

### 小矩阵 baseline（失败状态保留）

`20261117_phase1_smoke_eager`：长度 512/2048，C=1/4，seed=0，每条件 4 请求，输出固定 16 tokens。共 32 条测量请求，实际抢占 0。下表只能用于检查计时链路，**不是正式 p95 或优化结论**。

| 输入长度 | C | cold TTFT p50 / p95 ms | warm TTFT p50 / p95 ms | cold / warm 输出 tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 64.35 / 65.25 | 40.57 / 41.26 | 25.03 / 26.17 |
| 512 | 4 | 228.36 / 228.38 | 124.05 / 124.06 | 74.11 / 93.26 |
| 2048 | 1 | 242.41 / 242.85 | 38.86 / 39.16 | 20.46 / 28.60 |
| 2048 | 4 | 607.02 / 942.00 | 141.85 / 141.86 | 41.08 / 91.07 |

每格 n=4；p95 线性分位数及固定种子的 IID bootstrap 区间保存在汇总 JSON，但同波请求相关，区间不能证明尾延迟稳定。吞吐是输出 tokens / 对应测量波墙钟之和，包含 prefill/decode，不是纯 decode 或单请求速度。baseline GPU peak allocated 13.358 GiB、peak reserved 13.402 GiB、CPU RSS 最大约 1.549 GiB；设备占用与每波内存原始值另存，不能混为同一显存口径。

运行在最后一组 strict greedy 比较处失败：2048/C=4 中一条请求从首 token 开始不同，其余三条一致。之前 255/256/257/512/513/2049 边界、partial hit、共享前缀用例通过。保留 `status.json: success=false` 和 `failure.txt`，没有重写为成功；原计划该进程中的 profiler 未执行，后续单独完成。

### 分歧诊断

`20261117_cache_probe` 读取失败运行的四条原始输入，只比较相同 prompt 的第一个输出位置；保留全部 151936 维 logits 的压缩数组及请求记录。

- 分歧请求 cold 的 token 44、198 logits 均为 `11.3125`，argmax 按索引选 44。
- warm batch 对应值为 `11.3125`、`11.375`，选 198；全词表 max absolute difference 为 `0.109375`。
- warm single 再次并列 `11.375`、`11.375`，选 44。其他三个请求的首-token argmax 都一致。

这些观察支持 BF16 数值与形状敏感的解释；未选择事后容差宣告缓存语义完全通过。接下来应增加固定 token 路径的逐位置 logits/KV 校验，事先确定误差与并列值判据，再冻结标准 baseline。现有 strict greedy gate 不被静默放宽。

### 独立 prefill 成本与 trace（通过）

`20261117_prefill_profile`：C=1、仅生成 1 token，每长度先 2 个 cold/warm 预热对，再测 5 对不同的固定输入；全部首-token greedy 比较一致。正式计时关闭 profiler，之后额外采集一个 2048-token 冷 prefill trace。

| 输入 tokens | cold 实际计算 | warm GPU hit / 实际计算 | cold runner p50 ms | warm runner p50 ms |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 256 / 256 | 65.39 | 37.98 |
| 2048 | 2048 | 1792 / 256 | 249.61 | 40.67 |
| 8192 | 8192 | 7936 / 256 | 1220.07 | 57.51 |

8192 的 cold 由 4 个 2048-token chunk 组成，runner 成本为各 chunk 之和；warm 仍计算末尾 256 tokens，遵守上游约定。n=5 仅作成本初表，未声称正式 p95。单个 profiler 窗口中 `aten::mm` self CUDA 约 206.34 ms、FlashAttention 约 20.15 ms；其 trace 不参与以上计时，不把该窗口泛化为所有负载瓶颈。

### 真实 KV 布局搬运（包含失败与替代路径）

scratch KV 使用 `[2,36,pool_blocks,256,8,128]` BF16，与 Runner 轴顺序一致；源/目标 block IDs 交错分散，非把全层块误当成连续单次 memcpy。数据是随机 scratch tensor，**不是从模型生成 KV 的 CPU cache 恢复**。

`direct` 每个 K/V × layer × block 做连续 view 的 memcpy；`packed` 包含 GPU gather、连续 pinned copy 和 GPU scatter。内存分配与 view 描述准备在计时外，wall time 含发起调用和等待。每项 2 次预热、10 次测量；D2H 完成后才读 CPU buffer，H2D 完成后才复用 buffer。

| 完整块数 / tokens | 每方向 MiB | 路径 | D2H wall p50 ms | H2D wall p50 ms |
| ---: | ---: | --- | ---: | ---: |
| 1 / 256 | 36 | direct | 1.878 | 1.902 |
| 1 / 256 | 36 | packed | 1.852 | 1.862 |
| 4 / 1024 | 144 | direct | 6.676 | 6.939 |
| 4 / 1024 | 144 | packed | 6.591 | 6.811 |
| 8 / 2048 | 288 | direct | 13.550 | 14.001 |
| 8 / 2048 | 288 | packed | 13.020 | 13.435 |
| 32 / 8192 | 1152 | packed_chunked | 142.918 | 142.651 |

前六项来自 `20261117_kv_transfer`。每项输出日志前已通过 int16 bit view 的逐位 round-trip 及所有未写目标块仍为零的检查；该运行随后申请 32 块一次性 pinned tensor 时 OOM，整个 run 仍为失败。早期版本在全部完成后才写 summary，所以前六项由原始 JSONL 汇总，console 保留断言后的成功输出；后续脚本已改为每点立即写 summary。

最后一项来自独立的 `20261117_kv_transfer_bounded32`，只补跑缺失的 32 块：使用 288 MiB pinned staging、288 MiB GPU staging、1152 MiB pageable CPU scratch，包含额外 CPU copies 和逐 chunk 同步，round-trip 及未写块校验通过。它与前六项不是相同路径，不能用小块带宽线性外推。一次 OOM 不等于测出了 WSL pinned 内存硬上限；阶段 2 的 8 GiB CPU 容量建议不能直接解释为一次性分配 8 GiB pinned tensor。

这些微测未含 CPU cache 查表、LRU、真实活跃请求等待、保存策略和 suffix prefill 的端到端组合，因此没有给出 CPU 恢复收益或恢复/重算阈值。

## 复跑与工件

从仓库根目录运行，沿用现有虚拟环境。输出目录必须不存在，避免覆盖旧结果。

```bash
# 单元测试（8 项）；不需要 GPU
TORCHINDUCTOR_CACHE_DIR=/tmp/pegaflow-inductor TRITON_CACHE_DIR=/tmp/pegaflow-triton \
  .venv/bin/python -m unittest discover -s tests -v

# 本轮失败的小矩阵；保留 strict greedy gate，可能再次在相同位置失败
.venv/bin/python -m benchmarks.phase1 --output results/new_smoke --eager --profile

# 从已保存输入诊断；input-run 必须包含 2048/C=4 的四条 cold 请求
.venv/bin/python -m benchmarks.cache_probe \
  --input-run results/20261117_phase1_smoke_eager --output results/new_probe

# 仅需成本表和独立 trace 时，不必重跑已通过的边界用例
.venv/bin/python -m benchmarks.prefill_profile --output results/new_prefill

# 当前版本默认大于 8 块采用有界 staging；早期 OOM 版本在 source 快照中
.venv/bin/python -m benchmarks.kv_transfer --blocks 1 4 8 --output results/new_transfer_small
.venv/bin/python -m benchmarks.kv_transfer --blocks 32 --max-pinned-blocks 8 --output results/new_transfer32

# 只读分析现有结果，不启动 GPU 实验
.venv/bin/python -m benchmarks.summarize \
  --baseline results/20261117_phase1_smoke_eager \
  --transfer results/20261117_kv_transfer results/20261117_kv_transfer_bounded32 \
  --prefill results/20261117_prefill_profile --probe results/20261117_cache_probe \
  --output results/20261117_stage1_summary.json
```

原始记录位于 [results/](../results/)，汇总为 [20261117_stage1_summary.json](../results/20261117_stage1_summary.json)，测试原始输出为 [20261117_unit_tests.txt](../results/20261117_unit_tests.txt)。每个 run 的 console 位于 results 根目录的同名 `.console.txt`；trace 为 prefill run 中的 `prefill_trace.json.gz`。`requests.jsonl` 包含 token IDs、所有 token 时间及请求指标；`steps.jsonl` 包含实际调度 tokens、run/step 时间和使用块数；`waves.jsonl` 保存墙钟、吞吐和内存。

## 验收状态与下一步

- 已通过 8 项 CPU 测试：计时定义、块边界、共享引用/淘汰、抢占计数、关闭 metrics、cap 安全计算、admission、混合 greedy/采样。GPU 边界集、独立 prefill 首-token 对照及成功的搬运点证据分别保留。
- `compileall`、`git diff --check` 通过；对 32 条测量请求的时间顺序、命中/计算 tokens、输出长度、trace 中 CUDA kernel 和搬运校验状态做了只读一致性核验，见 `results/20261117_validation.json`。两次失败状态仍保持为失败。
- 未冻结：小矩阵包含严格 greedy 失败；尚无固定路径逐位置 logits 容差规范、CUDA Graph 正式对照、128-token 多 seed baseline、有 GPU 历史工作集容量压力的 trace、计时 instrumentation 开销评估和稳定的 p95 不确定性分析。
- 下一步先处理正确性验收规范，再跑少量多 seed/128 输出/CUDA Graph 的点并引入可复用前缀工作集压力。阶段 1 验收前不实现 CPU cache。
