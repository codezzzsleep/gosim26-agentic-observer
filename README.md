# Agentic Observer — 巡天观测智能体

GOSIM 2026 Agentic Observer Hackathon（https://create.gosim.org/survey26/）参赛作品，队伍 cloudbrain。

智能体在一个模拟的光纤光谱巡天里担任值班天文学家：每一步决定望远镜指向哪里、把每根光纤分给哪个目标、曝光多久、
用哪个观测程序，同时读公报和值班日志、判断仪器故障、管理计算和真实时间预算。
数值规划器负责每一次曝光决策；大模型负责值班日志的语义提取，以及夜间简报、故障裁决和赛季复盘四个环节，
任何一个环节失败都会退回规则，不会影响巡天继续进行。

## 比赛结果

决赛（隐藏任务卡 E–H）第 6 名：

| 总分 | E | F | G | H |
|---:|---:|---:|---:|---:|
| 31,680 | 36,969 | 29,502 | 27,216 | 33,035 |

## 最终提交

`agent/` 就是平台上评测的最终版本，文件与平台提交包逐字节一致（`agent/.env.example` 除外，它是开源时补的配置示例）。
平台修订号 `93d8609f6ba62e20ac8e6c1cf74b3aebe0e91192`；各文件的 SHA-256 见 `agent/SHA256SUMS`，可以这样核对：

```bash
cd agent && sha256sum -c SHA256SUMS
```

设计说明见 [`agent/README.md`](agent/README.md)，这一版相对前一版的改动见 [`agent/PROVENANCE.md`](agent/PROVENANCE.md)。

## 智能体概览

| 部分 | 文件 | 做什么 |
|---|---|---|
| 规划器 | `planner.py` | 每次决策评估上万个（指向，曝光时长）组合，按评测引擎相同的几何把目标映射到光纤格子；赛季计划给"光纤秒"定价，决定哪些目标现在拍完、哪些留到条件更好时 |
| 天空与仪器跟踪 | `tracker.py` | 在天气水平、仪器效率、是否有故障上做贝叶斯滤波，区分地震、仪器故障和圆顶临时关闭 |
| 方向性事件 | `hazards.py` | 公报里的地形遮挡、火箭发射、局部天气 |
| 指向偏差 | `pointing.py` | 用命中 / 未命中在偏差网格上做似然估计 |
| 账本与恢复 | `state.py` | 可重放的曝光账本，数据丢失后恢复每个目标的成绩 |
| 主循环与守卫 | `agent.py` | 协议、故障上报、CPU 节奏控制、真实时间守卫（预测评测引擎越跑越慢的耗时，必要时改为更少更长的曝光，避免在时间上限处被截断） |
| 值班日志 | `duty.py` | 大模型从公开的值班日志原文里提取故障、校准和更正，代码逐条校验来源行号、时区和时间区间后才转成行动 |
| 大模型环节 | `supervisor.py` `advisor.py` `llm.py` | 夜间简报、故障裁决、带工具调用的赛季复盘；限时等待、限流重试、多家服务自动切换，答案只在固定范围内生效 |

## 开发过程

| 版本 | 内容 | 平台成绩（正式赛 A–D 均分） |
|---|---|---:|
| v1 | 规则核心：规划器、赛季计划、跟踪器、故障上报、指向偏差、数据丢失恢复、CPU 节奏 | 31,148.84 |
| v2 | v1 + 空闲时段的圆顶探测与重新定价、独立的指向偏差估计、更紧的光纤边缘余量、地形遮挡按时段记录 + 第一版大模型环节 | 32,271.83 |
| v3 | v2 核心 + 不改决策的提速 + 真实时间守卫 + 按实测重做的大模型调用（暂停—等待—继续、限流重试、服务切换） | — |
| v5 – v8 | 在 v3 基础上：故障恢复、值班日志的大模型语义提取（`duty.py`）等，见 `agent/PROVENANCE.md` | — |

`lab/` 是比赛期间的本地闭环实验环境（截至 10 月 7 日的快照），v1–v3 在其中开发和验证：
用官方模拟器按种子造带天气的本地任务卡（84 张，分开发 / 调参 / 确认 / 封存四层），
跑分、配对统计比较、健康检查、上传前把关（含一个模拟各种故障模式的本地假模型服务），以及光纤时间去向等分析工具。
用法见 [`lab/README.md`](lab/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `agent/` | 最终提交的智能体（平台上运行 `python3 -u agent.py`，唯一依赖 numpy） |
| `lab/tools/` | 造卡、跑分、统计比较、把关、组装、分析工具 |
| `lab/agent/`、`lab/baselines/` | 开发过程中的核心：v3（工作副本）、v1、v2 |
| `lab/llm_stages/` | v3 时期的大模型环节（与核心分开存放，由 `tools/assemble.py` 组装） |
| `lab/cards/SETS.json` | 本地任务卡的配方；卡本身由 `tools/make_cards.py` 生成，不在仓库里 |
| `lab/results/` | 本地与平台成绩记录、Kimi 中转实测记录 |
| `lab/PROMPT.md`、`lab/UPDATE-*.md` | 当时写给队友本地 AI 助手的任务说明和更新记录 |

## 运行

需要 Python 3.10+ 和 numpy，以及主办方的官方示例包（本地裁判程序和任务卡，不在本仓库中）：

```bash
cd agent
pip install --target .deps -r requirements.txt
cp .env.example .env            # 可选：填一组模型服务的密钥；不填则纯规则运行
PYTHONPATH=.deps python3 <官方示例包>/runner/run_local.py --inherit-env \
    --card <官方示例包>/local-cards/L1 --agent "python3 -u agent.py" --agent-cwd .
```

本地实验环境：把官方仓库放到 `lab/hackathon-survey26/`、官方任务卡的公开输入解压到 `lab/cards/official/` 后运行 `bash lab/setup.sh`。

## 许可与引用

本队编写的代码和文档采用 MIT 许可（`LICENSE`）。

主办方提供的比赛数据、任务卡、评测程序与示例项目按 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 提供（署名、非商业），
不在本仓库中，详见 `NOTICE.md`。使用本仓库中依赖这些材料的部分（例如 `lab/` 的造卡与跑分工具）时，请引用：

> GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/)

主办方将发布介绍本基准的正式文章，发布后请改为引用该文章（届时在此补上）。

---

## English summary

Entry of team cloudbrain to the GOSIM 2026 Agentic Observer Hackathon; 6th place in the final round (hidden cards E–H,
total 31,680). The agent runs a simulated multi-fibre spectroscopic survey: at every step it chooses the pointing, the
fibre assignments, the exposure time and the observing program, reads bulletins and duty-log notes, detects instrument
faults, and manages both its CPU budget and the real-time cap. A numpy planner makes every exposure decision; an LLM
extracts facts from the public duty log (validated against line citations, time zones and intervals before use) and
runs a nightly briefing, a fault verdict and a tool-calling season review, each confined to fixed ranges and falling
back to the rules on any failure.

- `agent/` — the final submission, byte-identical to the platform package (see `agent/SHA256SUMS`; `.env.example` added).
- `lab/` — the local closed-loop lab used during development (card generator, benchmark, paired comparison, pre-upload
  gate with a mock model service, analysis tools). The organizers' engine and task cards are not included; see `NOTICE.md`.

Code and documentation by the team: MIT. The organizers' data, task cards, evaluation programs and example projects
are provided under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and are not included. When using
them, cite GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/), or the organizers' paper on the
benchmark once it is published.
