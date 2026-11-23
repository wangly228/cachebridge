# 完整前缀缺口的两层回放

**66⅔% 重访档的 33 条未完整恢复请求中，27 条的首个缺块此前遇到首次写入额度，6 条的首个缺块曾在 CPU 发布、随后被 LRU 淘汰。** 前者导致 40,960 个历史前缀 tokens 重算，后者导致 9,216 个，总计 50,176 个。每条还需要执行 128-token 新尾部查询，该部分单列。上一轮正式三臂对照中，混合与全长负载各自 96 条重复前缀已全部连续命中。两种正式负载的整段 p95 插值位置都在首次填充。[逐条缺口及经历的事件](../results/20261123_prefix_replay/boundary/requests.jsonl) · [正式负载回放](../results/20261123_prefix_replay/formal/summary.json)

## 方法与校准

参考 PrefixSim 提交 `160d5f81724dd9ec28b10d6abb64615e6adbb332` 的请求级块轨迹回放及两种计数口径，即驻留块数、避免的 prefill 工作量。该版 PrefixSim 把首个 miss 后仍然驻留的块也计为计算节省。cachebridge 按同一请求从头连续命中的 GPU 和 CPU 块计算实际复用，首个缺口之后的常驻块单独记录。参考程序随附轨迹的块大小为 16 tokens，本轮采用原始输入 IDs、引擎的 256-token 块、原有哈希链和尾块处理。[PrefixSim 设计](https://github.com/cacheMon/PrefixSim/blob/160d5f81724dd9ec28b10d6abb64615e6adbb332/doc/design.md) · [回放源码](../benchmarks/prefix_replay.py)

离线运行公开引擎的 Scheduler 和 TieredBlockManager。每个冷周期新建 GPU/CPU 元数据，各波沿用前波状态。Scheduler 自行分配 GPU 物理块、执行 chunked prefill、入批、抢占和释放。正式请求以归档 greedy token IDs 驱动序列状态，搬运器模拟 slot 就绪。后台保存仍经过引擎原有的预留、等待和整批发布调用点。事件记录描述调度器可见的先后关系。每请求的 TTFT 取自 3090 原始日志，反事实路径只统计元数据和实际调度的 tokens。边界运行文件没有保存 namespace 原字节，回放用固定独立盐重建等价的单模型键链，报告内的块摘要随该盐生成。[边界配置](../results/20261123_revisit_boundary/engine.json) · [回放与实现摘要](../results/20261123_prefix_replay/boundary/summary.json)

| 归档 overlap 32 | 正式请求 | 波数 | 调度步 | 核验结果 |
| --- | ---: | ---: | ---: | --- |
| 三档重访比例 | 432 | 108 | 14,187 | 每步和逐请求计数一致 |
| 混合及全长正式负载 | 432 | 108 | 14,136 | 每步和逐请求计数一致 |

逐步核对 prefill/decode 类型、入批请求 ID 和顺序、调度 tokens、活跃 GPU 块数。逐请求核对初始 GPU/CPU 命中、实际恢复、实际计算、H2D/D2H 块数及批次、抢占和失败。每波结束时在途写入清空，CPU 常驻块数与累计淘汰均等于实测。基线重组出的输出 ID 与冻结记录相同，模型计算的数值正确性沿用上轮真机同形状验收。[边界逐步实测](../results/20261123_revisit_boundary/steps.jsonl) · [正式逐步实测](../results/20261123_overlap_formal/steps.jsonl)

## 33 条请求的第一个缺口

4 个种子，各自 9 波、每波并发 4，波和块序号均从 0 开始。该档前 3 波填入互异的目标前缀，后 6 波重访，96 条重访全部 GPU 零命中。其中 63 条完整恢复，该分母涵盖第一次和第二次重访。

| 首缺块经历 | 未完整恢复 | 首缺位置 | 受影响波 | 因缺口重算的历史前缀 |
| --- | ---: | --- | --- | ---: |
| 额度触发，未写入 CPU | 27 | 26 条在块 0，1 条在块 4 | 波 1 的 11 个首缺块延迟准入，在波 4 重访时缺失。波 2 的 16 个在波 5 重访时缺失 | 40,960 tokens |
| 已经写入，后被 LRU 淘汰 | 6 | 全在块 0 | 波 0 预留并发布。在波 5 为其他键腾出 slot，波 6 重访时缺失 | 9,216 tokens |

其中一条请求虽然首块缺失，块 4–7 仍在 CPU，它的 1,024 个滞留 tokens 全部重新计算。逐条记录含 seed、原始 seq_id、FIFO 位置、首缺块摘要、首次访问波、触发拒写或淘汰的请求、替换键，以及缺口之后仍驻留的块。全事件流还可查预留时的 GPU ID 与 CPU slot、发布和 GPU 物理页再次分配。[完整事件链](../results/20261123_prefix_replay/boundary/events.jsonl) · [逐请求记录](../results/20261123_prefix_replay/boundary/requests.jsonl)

66⅔% 档每臂 144 条请求，实测 p95 913.92 ms。支撑插值的两条记录均是首次访问，种子 1803 波 2 的 905.48 ms 和种子 1801 波 0 的 915.41 ms。最慢 8 条中有 5 条属于波 5 的重访。如果固定首次访问的全部实测延迟，把每条重访的延迟都降到 0，再重新排序，p95 为 884.44 ms，极限降幅 3.23%。这一秩次敏感性检查还免去了恢复和尾部工作，用来识别哪些样本能移动分位点。原来两种正式负载在相同检查下 p95 均保持原值。[各档边界分位点与尾部请求](../results/20261123_prefix_replay/boundary/summary.json) · [正式分位点](../results/20261123_prefix_replay/formal/summary.json)

## 保持容量的离线策略比较

三组 GPU 均为 36 块、CPU 均为 64 slot。每个反事实独立重跑调度，使用相同的入队顺序和冻结输出串。首用额度放宽策略在离线实验中允许首次写满 64 个空 slot，满后继续执行原有复用准入。叶优先 LRU 维持 32 块准入和原有写入时点，当 CPU 满时优先逐出没有常驻后代的最旧节点。两种策略均仅访问当时已观察的请求和缓存状态。

| 重访占比 | 策略 | 完整重访 / 重访请求 | D2H 写块 | H2D 恢复块 | 实际计算 prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 0% | 现行 / 首用 64 / 叶优先 | 0 / 0 | 128 / 256 / 128 | 0 / 0 / 0 | 各 239,616 |
| 33⅓% | 现行 | 21 / 48 | 240 | 128 | 206,848 |
| 33⅓% | 首用 64 | 39 / 48 | 276 | 236 | 179,200 |
| 33⅓% | 叶优先 | 21 / 48 | 240 | 128 | 206,848 |
| 66⅔% | 现行 | 63 / 96 | 292 | 380 | 142,336 |
| 66⅔% | 首用 64 | 84 / 96 | 292 | 508 | 109,568 |
| 66⅔% | 叶优先 | 63 / 96 | 288 | 384 | 141,312 |

扩大首次写入额度能在重访密集时改变写入时机，66⅔% 档写入块数相同，多恢复 128 块。放到正式混合与全长负载里，两种负载的 D2H 都由 384 升至 768 块，实际计算仍分别为 251,904 和 221,184 tokens。正式集内改变叶节点淘汰顺序也没有改变恢复量。各方案的 GPU TTFT 仍需真机测量。[重访首用 64](../results/20261123_prefix_replay/all_free/summary.json) · [重访叶优先](../results/20261123_prefix_replay/leaf_lru/summary.json) · [正式首用 64](../results/20261123_prefix_replay/forced_formal_all_free/summary.json) · [正式叶优先](../results/20261123_prefix_replay/forced_formal_leaf_lru/summary.json)

## 复跑与输入新轨迹

从公开仓库根目录使用已有环境，输出选新目录。两种归档格式都可用于校准基线或比较策略。各运行目录含原始命令、输入/输出 SHA256 及引擎源码摘要。

```bash
py=../PegaFlow/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.prefix_replay --input results/20261123_revisit_boundary --output results/revisit_replay_repeat
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.prefix_replay --input results/20261123_overlap_formal --policy all_free --output results/formal_replay_64_repeat
```

也可以提供自己的 JSONL，一个物理行表示一波，每个 cycle 从空缓存开始，同一 cycle 内各波按文件出现的先后顺序运行。下面是两波简短的格式样例。实际评估前缀复用，输入需覆盖至少一个完整的 256-token 块，并保留尾查询。每条请求提供非空输入和冻结输出 IDs。传入新轨迹时只读取指定旧运行的 `engine.json` 作为容量与调度配置，输出包含两层命中、计算量和管理事件。

```jsonl
{"cycle": 0, "wave": 0, "requests": [{"input_token_ids": [12, 13, 14], "output_token_ids": [20, 21]}]}
{"cycle": 0, "wave": 1, "requests": [{"input_token_ids": [12, 13, 15], "output_token_ids": [22, 23]}]}
```

```bash
PYTHONDONTWRITEBYTECODE=1 "$py" -m benchmarks.prefix_replay --input results/20261123_revisit_boundary --transcript my_waves.jsonl --policy leaf_lru --output results/my_replay
```

## 按实验臂校准冷请求打包

使用 `--arm` 选择 CPU 缓存臂。三臂实验从 `engine.json` 的 `arms` 逐臂读取搬运方式、CPU 容量和冷请求打包开关。双臂搬运实验读取 `modes` 与 `pack_cold_by_arm`，旧归档使用运行配置。2026-11-23 冷请求打包筛选的 candidate 臂开启后台搬运和冷请求打包，36 条正式请求、9 波、1,169 个真实调度步与 CPU 回放的对应计数全部一致。16 次重访均取回所需的完整前缀，GPU 和 CPU 均有贡献，两种负载合计 H2D 56 块、D2H 64 块。[校准记录](../results/20261123_prefix_replay/cold_pack_screen/summary.json) · [逐请求计数](../results/20261123_prefix_replay/cold_pack_screen/requests.jsonl) · [此次运行脚本](../results/20261123_prefix_replay/cold_pack_screen/prefix_replay.py)

```bash
PYTHONDONTWRITEBYTECODE=1 ../PegaFlow/.venv/bin/python -m benchmarks.prefix_replay --input results/20261123_cold_pack_screen --arm candidate --output results/cold_pack_screen_replay_repeat
```

旧边界、旧正式归档及其四个策略对照共六次运行的脚本快照与各自汇总中的 SHA256 一致。新版将当次脚本复制到输出目录。[旧运行脚本](../results/20261123_prefix_replay/prefix_replay_v1.py)
