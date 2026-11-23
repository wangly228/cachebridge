# BF16 前缀复用核验协议 v1（2026-11-18）

本协议在本轮逐位置 GPU 核验运行前写入文件，并随实验保存 SHA256。不改引擎采样、GPU cache 或既有 strict greedy baseline gate。2026-11-17 的首-token 分歧已知，因此旧输入只作回归样本；新的 seed=20261118、20261119 输入另记 held-out，不能把旧输入称为独立校准集。

## 输入与路径

- 使用旧失败组的 4 条 2048-token prompt 及其 **cold 输出 token IDs**；重放 16 个输出位置。
- 新增 4 条 seed=20261118、长度 2048 的输入，以及 1 条 seed=20261119、长度 8192 的输入。新输入先以 cold greedy 生成 16-token 参考路径，后续所有比较强制使用这条路径。
- 冷/热比较在同一 prompt、同一输出位置、同一历史输出 token 下进行；不能在生成分叉后比较不同输入的 logits。
- 每组执行 cold、warm、warm_repeat、warm_after_roundtrip。后两者是新增的精确对照，非性能重复。全部固定 eager、BF16、block=256、KV cap=160、max batched tokens=2048。

## 必须精确成立的条件

1. 每次执行前校验已消费的历史 token 与冻结路径相同，记录对应的绝对 query position；Chunked Prefill 中间块不冒充输出位置。
2. warm 首次分配命中 `(prompt_len-1)//256` 个连续块；逐块校验 token IDs、前缀 hash 链和正 ref_count。
3. cold 计算结束时复制完整可复用 prompt 块作 GPU 快照。每次 warm 模型执行前后，复用块必须与快照 **逐位相同**；新计算的尾块不要求与不同形状的 cold 逐位一致。
4. warm_repeat 与 warm 在同一固定路径各位置的全部 vocabulary logits 必须逐位相同。
5. 等待所有请求结束，仅对 ref_count=0、hash/token 仍匹配的模型真实 KV 做 D2H→H2D 回原位诊断。一次只用 1 块（36 MiB）的 pinned scratch；先 D2H 完成再 H2D，H2D 完成再复用 buffer。不构建 CPU cache、不更改命中元数据、不覆盖活跃块。
6. D2H 后先校验 CPU 值，再将该空闲 GPU 块临时填 NaN 并确认变化，再做 H2D，防止无效复制被误判通过。round-trip 每块与 cold 快照逐位相同；随后相同 warm 路径的每个位置 logits 与搬运前 warm 逐位相同。

精确条件一旦失败就保留错误并停止该 run，不用浮点容差掩盖。

## 不同计算形状的数值筛查

令 BF16 epsilon=`2^-7`，每个输出位置的 `scale=max(1,max(abs(cold_logits)),max(abs(warm_logits)))`。预先采用以下工程筛查阈值：

- 所有 logits 有限。
- `max_abs_error <= 2 * epsilon * scale`。
- `L2(error)/L2(cold_logits) <= 4 * epsilon`；分母最小为 `1e-12`。
- 单独报告 argmax 是否一致、cold top2 间距，以及 warm 所选 token 在 cold 下距离最大 logit 的 regret。`regret <= 2 * max_abs_budget` 仅分类为近并列候选，不替代上述数值和精确条件。

阈值是公开的工程筛查范围，**不是 BF16 误差的数学上界，也不是由此宣布模型语义或阶段 1 完整验收**。旧样本已知，新的 held-out 样本规模也小。运行后不为通过而调阈值；若失败，记录位置、全量 logits 和差异，另行调查。旧 strict greedy 失败记录保持不变。

## 工件与解释

保存每个位置的全部 logits、实际 greedy 选择、强制使用的 token、query position、缓存命中、KV 检查次数、输入/输出路径、模型/配置与源码快照。logits 压缩归档，KV 仅保留逐位比较结果和每块摘要，避免保存数 GiB tensor。

检查过程会同步、复制和计算摘要，所以所有计时均只用于诊断，不加入 baseline 或成本表。完成此协议后，仍需单独验证 CUDA Graph、128 输出、多 seed/容量压力负载和计时开销，才能冻结阶段 1。
