# 今晚到最终交付

## 范围

主线是 milestone 的多模态扩展及 “Price the defenses”。缓存隔离边界属于 “Channels that survive salting” 的多模态部分。Evan 的文字恢复和共享 GPU 争用仍由他推进。

不把结果预写成“防御一定失效”。新计时定义可能改变效应量甚至结论，必须如实记录。

## 0. 免费准备，约 20–40 分钟

- 保留 `legacy/`，新结果只放独立 run 目录。
- 校验 SSE 首 chunk 和首内容时间、DONE、指标缺失处理、互补配对、图片哈希。
- 保存公开自然图片来源。默认 64 张豆叶照片：4 warmup、8 calibration、40 evaluation、12 performance；独立划分且无重复哈希。**这些照片来自一个领域，不代表多领域图像泛化。**
- 核对 Modal 认证、单价与预算。用户已选用 A100 40GB，所有新任务单 GPU 串行。

## A. 小试跑，优先完成

pilot 使用 4 张校准、4 张评估、2 张性能图片，仅 distinct salts，一次 repetition。attack 两个独立服务器会话，cost 三个会话；合计五次小服务器启动。启动开销高于请求开销。

通过条件是技术性的：流式输出完整、无请求失败、计时顺序正确、配对数符合计划、KV 指标有可解释的更新。**不以检测分数高低决定是否通过。**

失败时：

- OOM：先降低上下文到 2048、`max_num_batched_tokens` 到 2048、显存利用率适度调节；仍不行就停止，保留日志。不能悄悄更换模型并和旧模型结果混合。
- 指标延迟：检查 `/metrics` 更新周期。缺失/延迟不是 0 命中；修好后新建 run，不给坏记录伪造标签。
- SSE role chunk 与 content 差异大：继续使用 content 指标，旧 chunk 测量单列历史证据。
- 模型启动超过 10 分钟：停止并看日志，不再盲等/自动重试。

## B. 正式图片成员检测

40 evaluation + 8 calibration，shared/distinct 两种 salt 模式，3 次 repetition，每模式两次全新服务器会话。每张图片在互补会话中各接受一次 cached 和一次 unseen 条件。每会话每图片只有一次攻击探测。

随机化请求顺序、模式顺序、互补会话顺序。固定模型、分辨率和提示词。known-image membership 是威胁模型；不声称图片恢复。

预定主指标：first-content ROC AUC、AP、校准阈值的 TPR/FPR、同图 paired 延迟差。保留首 chunk 时间和 KV 计数供诊断。每个 repetition 单列；图片 ID 和 repetition 为重采样单位，不把所有请求当作独立样本。

## C. 独立复用成本

另用 12 performance 图片。每 repetition/mode 对冷、同租户复用、跨租户复用分别启动全新服务器，共 18 次启动。避免先测同租户把跨租户样本预热。

报告 cold−same、cold−cross 的同图配对延迟差及每次运行结果。只有 shared/distinct 下的图片复用成本，不叫“完整防御的代价”。processor cache 关闭只是消融，不能说是完整租户隔离。今晚不测吞吐，不画虚构 throughput frontier。

B+C 总计 30 次短服务器启动。如果 pilot 证明启动太慢，先运行 B，把 C 改为单独配置/下一轮完成；不要在两小时截止时硬跑半个设计。

## D. 机制直接观测（可选，不阻塞交付）

见 `MECHANISM.md`。先验收补丁与 vLLM 0.11.0 路径；一小时内无法验证就结束。未观测到的机制明确写推断。计数补丁不得装入正式延迟测量。

## 最终交付

1. `requests.jsonl`、完整 config/meta/数据集 manifest、服务器日志。
2. 新真实图片结果图（包括重复性/区间）、独立评估 TPR/FPR。
3. 图片隔离效果与延迟复用收益表（带同模型硬件版本）。
4. 历史文字/图片/音频证据索引，版本分开；旧恢复报告存在旧批次，不当成本主线结果。
5. Threats to Validity：一个叶片图片域/模型/版本、localhost、三次 session、无任意图像重建、无全栈修复、KV 计数不等于 encoder 命中。

## 结论规则

- 有区分信号且 KV 统计一致：支持“KV 隔离不能覆盖被测多模态计时路径”，具体缓存归因需 D 或消融支持。
- 没有区分信号：报告估计值与区间，不能宣布“绝对无泄露”；对照旧 chunk 测量和自然图/噪声图差异。
- 单次结果翻转：报告 run-level 差异，排查来源，不挑选最好一次作为结论。
