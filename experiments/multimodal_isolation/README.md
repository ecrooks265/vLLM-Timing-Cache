# pcache 多模态隔离实验启动包

研究问题：`cache_salt` 隔离 KV 前缀之后，图片请求是否仍会跨租户复用处理结果？这种复用能否用于已知图片的成员检测，又节省多少首内容延迟？

**正式图片实验已完成；pilot 只用于技术校验。公开版包含原始测量、源码快照及分析结果。**

**本次 A/B/C 已完成：见 `docs/FINAL_RESULTS_zh.md`。下列启动命令用于重跑，不需要为了查看结果再租 GPU。**

## 从这里开始

```bash
cd experiments/multimodal_isolation
# 建议在独立虚拟环境安装；已有 Python 环境可复用。
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_images.py
python -m unittest discover -s tests -v
python scripts/run_stage.py pilot --check  # 免费，只检查预算
python scripts/run_stage.py pilot          # 会启动收费 A100，小试跑
```

上次已登录 Modal 的机器可以沿用登录；新机器先运行 `modal token new`，本人完成登录。
已有 `pcache-out` volume 是前提，本包不会自动新建 volume，也不需要第三方 API 密钥。

运行日志：`results/launch-*.log`。成功后自动下载到 `results/runs/<run_id>/`。

```bash
python scripts/analyze.py results/runs/<pilot-run-id>
# 仅技术校验通过才会生成 pass=true 的 results/pilot_gate.json。
# 先人工检查 REPORT.md、计数器、错误、耗时；不用因为 pilot 有正信号才继续。
python scripts/run_stage.py main --check
python scripts/run_stage.py main
python scripts/analyze.py results/runs/<main-run-id>
```

如果会话中断或运行失败，结果可能已在远端。不要盲目重跑：

```bash
modal volume ls pcache-out multimodal_starter/runs
modal volume get pcache-out multimodal_starter/runs/<run-id>/ results/runs/<run-id>
modal app list
# 仅停止本研究对应的 app-id，不能停止其他项目。
modal app stop <this-study-app-id>
```

## 文件导航

| 文件 | 用途 |
|---|---|
| `docs/PLAN_zh.md` | 问题、阶段、停止条件和最终交付 |
| `docs/PROTOCOL.md` | 配对、随机化、计时、分析定义 |
| `docs/BUDGET.md` | 单价、预算预留及其限制 |
| `docs/MECHANISM.md` | D 阶段的直接观测方案、现有补丁的检查点 |
| `docs/FINAL_RESULTS_zh.md` | 本次完整实测结果、结论范围与衔接 |
| `docs/DEFENSE_BOUNDARY.md` | 文字／图片／音频的证据边界 |
| `configs/pilot.json`, `configs/main.json` | 小试跑和正式配置 |
| `scripts/prepare_images.py` | 公开叶片照片下载、归一化、去重与 SHA256 清单 |
| `src/protocol.py` | 流式计时与分组逻辑 |
| `src/runner.py` | A/B/C 顺序实验，自动清理服务器 |
| `modal_study.py` | 单 A100，无自动重试，无自动升级 GPU |
| `scripts/run_stage.py` | 串行预算预留、调用和下载结果 |
| `scripts/analyze.py` | 技术门槛、独立评估、区间、图及报告 |
| `legacy/` | 原始 pcache 的代码、数据和研究说明；省略终端日志，不混入新结果 |

本包不包含文字恢复、候选搜索、模板后缀扫描、GPU 争用实验。没有账户总费用的程序化硬限额；本地预算保护只管理经本包启动的任务，详见预算文档。

新机器如果没有实验 volume，先运行 `modal volume create pcache-out`。查看或重新分析现有结果不需要 GPU；`scripts/run_stage.py pilot/main` 会启动收费计算，必须主动执行。

Public-release provenance and exclusions: see `docs/PUBLIC_RELEASE.md`.
