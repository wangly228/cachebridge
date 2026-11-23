# 3090 上的 INT4 历史前缀缓存

**INT4 值得用于 CPU 页池或 PCIe 受限、历史前缀反复重放且容许输出变化的已验证负载。通用默认仍选 BF16。** 同一块实际节省 71.875% 主存和双向传输载荷。六种子下，同 slot 的混合淘汰重放 p95 降低 34.46%；同页池字节的容量重访 p95 降低 82.73%。含冷填充的混合整段仅改善 0.76%。独立两跳检索在 64-token 回答预算下从 13/18 降到 7/18，未通过预设质量线。单跳检索的 18 个样本通过该类判据。对必须复现原文本或有紧输出预算的多跳任务，保留 BF16。[性能汇总](../results/20261123_int4_prefix_formal/summary.json) · [质量汇总](../results/20261123_int4_quality_formal/summary.json)

## 启用方式和实际数据路径

在常规 `LLM` 入口设置 CPU slot 数和格式。省略 `cpu_prefix_cache_format` 或设为 `"native"` 就使用模型原生格式，Qwen3-4B 在本机为 BF16。原生格式下设 `cpu_prefix_cache_blocks=0` 关闭 CPU 层。下面是正式实验的同 slot 候选配置。

```python
from pathlib import Path
from nanovllm import LLM

llm = LLM(
    str(Path.home() / "models/huggingface/Qwen3-4B"),
    max_model_len=9216, max_num_batched_tokens=2048, max_num_seqs=8,
    max_kvcache_blocks=36, gpu_memory_utilization=0.8, enable_metrics=True,
    cpu_prefix_cache_blocks=64, cpu_prefix_cache_format="int4",
    cpu_prefix_admission="reuse", cpu_prefix_first_use_blocks=32,
    cpu_prefix_batch_writes=True, cpu_prefix_transfer_mode="overlap",
    cpu_prefix_pack_cold=True,
)
```

Qwen3 把经过归一化和 RoPE 的 K 与投影后的 V 写入每层的 BF16 GPU 页。完整 prompt 块获准保存时，一个 Triton kernel 从同一物理块跨 36 层直接读取 K/V，编码到一个连续的 GPU 字节缓冲。D2H 把这段字节传入锁页缓冲，再复制到可分页 CPU slot。恢复沿相反方向传压缩字节，解码 kernel 直接写入独占的 BF16 目标页。带前缀的 prefill 与 decode 都由原有 FlashAttention 读取这些 BF16 页。GPU 活跃 KV 量化、扩展活跃并发和每步 decode 加速未测。[模型写入](../nanovllm/models/qwen3.py) · [attention 读写](../nanovllm/layers/attention.py) · [搬运](../nanovllm/engine/kv_transfer.py) · [编解码](../nanovllm/engine/int4_cache.py)

K、V 各自按 token、头和连续 32 个坐标分组，保存 BF16 绝对最大值。临时以 FP32 计算 `q = round_even(7 × x / absmax)`，使用 −7 至 7 的 15 个整数值，全零组取零。编码加 8，偶数坐标占字节低四位。先写全块打包值，再紧接所有组的最大值。恢复计算 `(q / 7) × absmax` 并写 BF16。该开关限单卡、BF16 和头维度为 32 的整数倍。格式版本进入模型缓存 namespace，BF16 的旧 namespace 仍可使用已校准的成本表。本轮对照均在 CPU 命中时恢复，INT4 成本选路表未校准。[配置](../nanovllm/config.py) · [命名空间与提交顺序](../nanovllm/engine/cpu_prefix_cache.py)

## 参考实现与实验取舍

参考 Alberto-Codes/turboquant-vllm 提交 `7b56dc97bb7aefd9bae684b40607b87290159b6b`。其 Hugging Face `CompressedDynamicCache` 对 K、V 可分别选位宽，按向量保存范数，归一化和旋转后用 Lloyd–Max 码本索引。128 维、单个 K 或 V 向量在 4 位模式占 64 字节索引加 4 字节 FP32 范数；3 位模式的每个索引仍占一个 uint8，合计 132 字节。普通读取路径仅解码新 token，然后拼接到每层保留的解压 K/V；融合模式可直接交出压缩数组。单看压缩数组的统计会漏掉普通读取路径的常驻副本。本机实现独立采用分组对称量化和自己的页布局。[参考缓存源码](https://github.com/Alberto-Codes/turboquant-vllm/blob/7b56dc97bb7aefd9bae684b40607b87290159b6b/src/turboquant_vllm/kv_cache.py) · [参考量化器](https://github.com/Alberto-Codes/turboquant-vllm/blob/7b56dc97bb7aefd9bae684b40607b87290159b6b/src/turboquant_vllm/quantizer.py)

- 上轮测得每个 BF16 历史块 36 MiB，混合淘汰重放可以完整恢复，整段 p95 则常落在冷填充。本轮先压缩 CPU 驻留与双向 PCIe 载荷，同时观察重放和整段。[上轮验收](STAGE2_COLD_PACK.md)
- 种子 3400 的同 slot 筛查中，混合淘汰重放 4 条/臂的 p95 从 167.00 降到 103.18 ms，整段 20 条/臂从 827.90 到 822.02 ms。于是保留同 slot 对照，再以相同页池字节实测容量压力，正式计时关闭 profiler。[探索汇总](../results/20261123_int4_prefix_screen/summary.json)
- 最初 12-token 输出窗口把两臂的四个回答都截在记录编号处。用同一探索输入放宽到 64 tokens，EOS 前独立评分，两个格式各找对 4/4 个码，四条措辞均有变化。金标准数字的分布过于集中，于是增加跨记录两跳检索，并沿 BF16 自然回答比较同输入、同位置的完整 logits。两跳探索的 12288-token 两题两臂都答错，因而正式验收也分别列出 BF16 自身失败与量化损失。[原始截断结果](../results/20261123_int4_quality_screen/summary.json) · [同输入复测](../results/20261123_int4_quality_readout/summary.json) · [两跳探索](../results/20261123_int4_quality_crossref/summary.json)

## 实际字节

一个有效块含 256 tokens × 36 层 × 8 个 KV 头 × 128 维 × K、V，共 18,874,368 个标量。INT4 的数据占每值 4 位，计入分组最大值后为 **4.5 位/值**。下表以 MiB（2²⁰ 字节）计，分母包含全部层和 K、V。

| 分配或复制 | BF16 64 slots | INT4 64 slots | BF16 18 slots |
| --- | ---: | ---: | ---: |
| 每块数值 / 量化元数据 / 块内填充 | 36 / 0 / 0 | 9 / 1.125 / 0 | 36 / 0 / 0 |
| 可分页 CPU slot 池 | 2304 | 648 | 648 |
| 两块锁页暂存，逻辑张量合计 | 72 | 20.25 | 72 |
| CPU 以上两项合计 | 2376 | 668.25 | 720 |
| GPU BF16 活跃页池 | 1296 | 1296 | 1296 |
| GPU 搬运暂存，一块 | 36 | 10.125 | 36 |
| 单块 D2H 或 H2D 的 CUDA memcpy 载荷 | 36 | 10.125 | 36 |
| 满容量有效历史 tokens | 16384 | 16384 | 4608 |

同块数的 CPU 页及单次传输节省 **71.875%**。同为 648 MiB 可分页页池，容量轨迹实际填到 64 和 18 个有效块，**3.56 倍**。若给 CPU 的页池加两块逻辑锁页暂存合计 720 MiB，按当前布局可放 69 个 INT4 块或 18 个 BF16 块，69 是容量算术，实机候选配置为 64。每块元数据起点、页步长与保存的数组字节数均可核对，块间填充为零。GPU 原 BF16 页在被驱逐前仍占页池，INT4 的 CPU 层和搬运暂存只保存压缩表示。真实驻留重叠、索引和分配器另外计量。[INT4 分配清单](../results/20261123_int4_prefix_formal/3501-int4_64/engine.json) · [BF16 分配清单](../results/20261123_int4_prefix_formal/3501-bf16_64/engine.json)

筛查后在一块已计算的真实 GPU 页上单独做一次保存和恢复。CUPTI 在 D2H、H2D 两个方向均观察到 BF16 的 37,748,736 字节及 INT4 的 10,616,832 字节。此次 BF16 的 GPU 聚集/D2H/H2D/散写为 95/1553/1530/95 µs，INT4 编码/D2H/H2D/解码为 70/430/419/59 µs。这是单次诊断。正式搬运量按实际成功复制块数乘该格式的 DMA 载荷计算，PCIe 链路协议开销未测。[BF16 传输事件](../results/20261123_int4_prefix_screen/3400-bf16_64/dma_events.json) · [INT4 传输事件](../results/20261123_int4_prefix_screen/3400-int4_64/dma_events.json) · [原始 INT4 profiler 轨迹](../results/20261123_int4_prefix_screen/3400-int4_64/dma_trace.json.gz)

在正式容量轨迹，两个 64-slot 臂各有最多 29 个 CPU 块同时也在 GPU 索引中，18-slot 臂最多 18 个。这些 BF16 副本已包含在 GPU 的 36 页中。INT4 编解码直接读写页和暂存，只需 kernel 内的临时值。首个种子预热后，整个进程的锁页分配器 `allocated_bytes.current` 为 BF16 的 128.049 MiB 和 INT4 的 32.049 MiB，包含分配器取整及引擎的其它锁页张量。64-slot 臂的 CPU 索引可达 Python 对象最大均为 606,216 字节，含可能与 GPU 索引共享的 token 对象；该数字不能直接加到 RSS 上。[逐波占用](../results/20261123_int4_prefix_formal/3501-int4_64/waves.jsonl) · [内存与输出差异审计](../results/20261123_int4_audit.json)

| 六种子最高值 | BF16 64 slots | INT4 64 slots | BF16 18 slots |
| --- | ---: | ---: | ---: |
| GPU 张量峰值 / 分配器 reserved 峰值，GiB | 9.042 / 9.111 | 9.017 / 9.086 | 9.042 / 9.111 |
| 推理波结束采样的最大 CPU RSS，MiB | 4039.49 | 2287.49 | 2382.22 |
| 进程生命周期 CPU RSS 高水位，MiB | 4739.78 | 4739.74 | 4739.89 |

生命周期高水位还包括模型加载。GPU 峰值随较小的搬运暂存下降约 25.88 MiB；三臂都实际分配 36 页 BF16 活跃 KV。[内存、命中与吞吐汇总](../results/20261123_int4_prefix_formal/summary.json)

## 六种子逐请求时延

种子 3501–3506，每个种子轮换三臂顺序。每波并发 4，每请求 16 个输出 tokens，三个臂都开启冷合批和后台写入。混合和容量工作集分别有 120、288 条/臂，共 1,224 条请求。下列 p95 单位为 ms，变化量以 INT4 为分子，括号内是完整种子配对重采样的 95% 区间。正式运行以不同进程预热再清空索引，无 profiler。[冻结输入、顺序与计时方法](../results/20261123_int4_prefix_formal/design.json) · [完整汇总](../results/20261123_int4_prefix_formal/summary.json)

| 工作集与范围 | 条/臂 | BF16 64 | INT4 64 | BF16 18 | INT4 比 BF16 64 | INT4 比同页池字节 BF16 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 混合整段 | 120 | 837.64 | 831.29 | 827.55 | −0.76%（−1.32 至 −0.26%） | +0.45%（−0.12 至 +0.91%） |
| 混合冷填充 | 24 | 871.92 | 840.56 | 868.94 | −3.60%（−4.15 至 −2.28%） | −3.27%（−3.68 至 −1.34%） |
| 混合无复用 | 48 | 827.76 | 826.87 | 825.54 | −0.11%（−0.54 至 +0.78%） | +0.16%（−0.06 至 +2.51%） |
| 混合淘汰重放 | 24 | 177.07 | 116.05 | 529.19 | **−34.46%（−40.10 至 −31.41%）** | −78.07%（−79.95 至 −70.43%） |
| 混合 GPU 热重放 | 24 | 74.72 | 77.70 | 99.36 | +3.99%（−0.31 至 +4.71%） | −21.80%（−25.03 至 −21.51%） |
| 容量整段 | 288 | 555.85 | 556.46 | 562.16 | +0.11%（−0.13 至 +1.88%） | −1.01%（−1.28 至 +0.64%） |
| 容量延迟重访 | 96 | 145.32 | 97.14 | 562.47 | −33.15%（−40.05 至 −28.52%） | **−82.73%（−83.55 至 −82.69%）** |

| 工作集 | 格式与 slots | GPU 命中 / CPU 恢复 tokens | 实算 prompt tokens | D2H / H2D 块 | D2H / H2D GiB | 重访完整恢复 | 整批输出 tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 混合 | BF16 64 | 36864 / 36864 | 125952 | 192 / 144 | 6.750 / 5.063 | 24/24 | 81.48 |
| 混合 | INT4 64 | 36864 / 36864 | 125952 | 192 / 144 | 1.898 / 1.424 | 24/24 | 83.03 |
| 混合 | BF16 18 | 36864 / 25088 | 137728 | 154 / 98 | 5.414 / 3.445 | 14/24 | 76.94 |
| 容量 | BF16 64 | 98304 / 98304 | 135168 | 384 / 384 | 13.500 / 13.500 | 96/96 | 130.42 |
| 容量 | INT4 64 | 98304 / 98304 | 135168 | 384 / 384 | 3.797 / 3.797 | 96/96 | 136.82 |
| 容量 | BF16 18 | 98304 / 24576 | 208896 | 396 / 96 | 13.922 / 3.375 | 24/96 | 106.95 |

同 slot 的混合重放里，两臂从相同前缀恢复相同的 144 块，含主存复制、H2D、GPU 写回及同步的平均每块恢复耗时从 3.984 降至 1.339 ms。相同页池字节的容量轨迹里，INT4 多恢复 288 块，少重算 73,728 个 prompt tokens，H2D **总量反而增加 12.5%**。单块节省与工作集总流量需要分别解释。各臂传输失败、抢占均为 0。两项分段时延门槛及三项整段/无复用退化上限全部通过。[逐请求输入、首 token 时间和传输量](../results/20261123_int4_prefix_formal/3501-int4_64/requests.jsonl) · [离线验收](../results/20261123_int4_audit.json)

## 独立长上下文质量

正式种子 3201、3202。每条记录含一个随机六位验证码。单跳给出记录 ID，两跳给出票据 ID，需要先在长前缀另一处查到对应记录。每个长度覆盖三个深度，每个种子和位置上的两类题共用目标值。单条请求先填充真实 BF16 前缀，然后分别使用原始 GPU 页、从 CPU 完整恢复的 INT4 页，计算相同的尾部查询。36 题均真实恢复全部 2048、8192 或 12288 个历史 tokens。独立采集自由 greedy 输出、金标准数字的同位置 logits、沿 BF16 自然回答的同位置 logits。质量诊断单独用 52 个 GPU 块及 48 块首次额度，确保最长样本能整段驻留和恢复。完整配置与原始回答均已保留。[质量设计](../results/20261123_int4_quality_formal/design.json) · [逐题输入、输出、误差](../results/20261123_int4_quality_formal/cases.jsonl)

| 有效历史长度 | 单跳 BF16 → INT4 | 两跳 BF16 → INT4 | 两跳丢失的 BF16 正确答案 |
| --- | ---: | ---: | ---: |
| 2048 tokens | 6/6 → 6/6 | 6/6 → 5/6 | 1 |
| 8192 tokens | 5/6 → 6/6 | 4/6 → 1/6 | 3 |
| 12288 tokens | 5/6 → 6/6 | 3/6 → 1/6 | 2 |
| 合计 | 16/18 → 18/18 | 13/18 → 7/18 | 6 |

64 个输出 tokens 内，36 题 BF16 答对 29 题，INT4 答对 25 题，有 6 题失去正确答案、2 题新增正确答案。两臂都没有严格仅输出六个数字的样本；EOS 前全文仅 4/36 相同。单跳观察到的两条新增正确答案样本不足以证明量化提高任务质量。六种子性能输入上的同 slot 逐 token 对照也有差异，混合淘汰重放首 token 有 1/24 不同，16-token 序列有 15/24 不同；容量重访分别为 6/96 和 55/96。[任务得分及错误明细](../results/20261123_int4_quality_formal/summary.json) · [同输入 greedy 差异](../results/20261123_int4_audit.json)

| 同位置数值对照 | 单跳 | 两跳 |
| --- | ---: | ---: |
| 金标准数字的 top-1 一致 | 108/108 | 97/108 |
| 数字的平均 ΔNLL，nat/token | −0.004 | +0.044 |
| 自然回答 top-1 一致 | 382/398 | 551/576 |
| 自然回答平均 KL / ΔNLL，nat/token | 0.148 / +0.139 | 0.113 / +0.120 |

每个位置均比较原词表的 151,936 个 logits，共 1,190 个配对位置。金标准位置最大绝对差 21.25，最大相对 L2 为 1.099。每题取真实前缀的首、中、末三块比较重建 K/V，108 块 × 36 层。K、V 的逐层块相对 L2 中位数分别为 0.1320、0.1123，最大 0.1742、0.1506，最小余弦相似度 0.9849、0.9888。两跳的数字 top-1 一致率 89.81%，低于预设的 95%，两跳及最长档也均超过允许丢失一题的上限。自然回答分布的平均误差达标。[逐层误差、完整 logits 与原输入](../results/20261123_int4_quality_formal/) · [冻结线逐项判定](../results/20261123_int4_audit.json)

**输出窗口诊断。** 在全部 13 道任一格式失分的正式题上，用常规 `LLM.generate`、相同输入与原始/恢复页，将输出上限改为 192，并在 EOS 停止。26 条路径都逐 token 复现原有输出至 EOS 或第 64 位，且在 192 以内结束。原本 BF16 答对而 INT4 失分的 6 道两跳题全部取回正确码；其 BF16 完整回答长 47–59 tokens，INT4 长 81–140 tokens。所有原短窗口失分中，BF16 的 7 题有 2 题转为正确，INT4 的 11 题有 8 题转为正确，余下 3 个 INT4 错误均已到 EOS。这些是失败子集的事后复查，正式 64-token 判定按冻结条件计分。[复查规则](../results/20261123_int4_answer_length/design.json) · [逐题答案与长度](../results/20261123_int4_answer_length/cases.jsonl) · [复查汇总](../results/20261123_int4_answer_length/summary.json)

例如，种子 3201 的 8192-token、10% 深度两跳题，正确码为 `940896`。INT4 的第 64 个输出位置只能读到码的前三位，完整回答在 81 tokens 时结束并给出正确码。相同种子的 12288-token、50% 深度题正确码为 `142939`，即使延长窗口，BF16 给出 `683000`，INT4 给出 `159785`。本机质量诊断记录了 Torch 编译形状数量达到上限后回退的日志，配对时使用相同位置、相同查询形状；该诊断的墙钟未用于性能结论。[正式输出](../results/20261123_int4_quality_formal/cases.jsonl) · [扩窗输出](../results/20261123_int4_answer_length/cases.jsonl) · [诊断日志](../results/20261123_int4_quality_formal.console.txt)

本轮支持在与已测单跳查找相近、可以接受措辞变化的前缀重放上按需启用。无复用段同 slot 的变化区间跨零，GPU 热重放的点估计上升。两跳且回答窗口紧、需要逐 token 复现、或尚无业务质量样本时沿用原生格式。若扩至另一类任务，完成条件是先用该类独立长上下文样本和实际回答窗口复核正确答案丢失数，再在其真实到达序列比较端到端 TTFT。其它模型、超过 12288 tokens 的质量、更长多轮历史连续派生和活跃 GPU KV 量化未测。

## 正确性、复跑与范围

54 项测试通过。已知 signed 索引、半字节顺序、零组与 BF16 最大值逐项核验。真实 CUDA 后台写入三个非相邻块，跨 K/V、层、头和不整齐的 kernel tile，恢复值与独立 PyTorch 参考逐元素相同，重复编码得到相同字节。覆盖旧路径的 BF16 无损往返、完整块和尾查询、部分连续命中、共享与抢占、CPU 淘汰、在途 slot、异常清理及先完成再发布。新格式复用相同的块管理和同步协议。[完整测试日志](../results/20261123_int4_final_tests.txt) · [新格式测试](../tests/test_int4_cache.py) · [共享和在途保护](../tests/test_overlap_writes.py)

Git 仓库保留逐请求记录、逐层误差和质量得分。52 个原始张量留在实验机，可用下列 GPU 命令重新生成正式输入对应的完整词表 logits 和 KV 样本，再离线复核。原始采集文件的 SHA-256 见[张量清单](../results/20261123_int4_tensors.sha256)。

从 cachebridge 根目录运行，GPU 实验写入新的目录。每个正式目录都有运行命令、代码快照、工作树 diff、模型文件 SHA-256、revision、包版本和原始样本。性能原始数据位于各个种子和格式的子目录，包含逐请求、逐步及逐波日志。质量 NPZ 中 logits 和 K/V 小样的 int16 为 BF16 的原始位。模型 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`，PyTorch 2.10.0+cu128、Triton 3.6.0、FlashAttention 2.8.3，单张 3090 24 GB，WSL2。[正式环境](../results/20261123_int4_prefix_formal/environment.json) · [冻结判据](INT4_PREFIX_PROTOCOL.md)

```bash
export PYTHONDONTWRITEBYTECODE=1
py=../PegaFlow/.venv/bin/python
"$py" -m unittest discover -s tests -v
"$py" -m benchmarks.int4_prefix --stage screen --output results/int4_timing_pilot_repeat
"$py" -m benchmarks.int4_prefix --stage formal --output results/int4_timing_repeat
"$py" -m benchmarks.int4_quality --stage screen --output results/int4_quality_pilot_repeat
"$py" -m benchmarks.int4_quality --stage formal --output results/int4_quality_repeat
"$py" -m benchmarks.int4_answer_length --input results/int4_quality_repeat --output results/int4_length_repeat
"$py" -m benchmarks.int4_prefix --summarize --output results/int4_timing_repeat
"$py" -m benchmarks.int4_audit --timing results/int4_timing_repeat --timing-pilot results/int4_timing_pilot_repeat --quality results/int4_quality_repeat --quality-pilot results/int4_quality_pilot_repeat --output results/int4_audit_repeat.json
```
