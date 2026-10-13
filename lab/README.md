> **开源版说明（2026-10-13）。** 这是比赛期间本地闭环实验环境的快照（截至 10 月 7 日），只含本队自己写的代码和记录。
> 最终提交的智能体不在这里，在仓库根目录的 `agent/`。这里的 `agent/` 是开发中的 v3 核心，`baselines/v1`、`baselines/v2` 是平台上评过的两个版本的核心。
>
> 以下内容属于主办方（CC BY-NC 4.0），**没有包含**，需要自己准备后放到对应位置：
>
> - 官方仓库 / 示例包（评测引擎、本地裁判程序）→ `lab/hackathon-survey26/`，或用环境变量 `SURVEY26_REPO` 指向它；
> - 官方任务卡 ZIP 的公开输入（A–D、α–δ）→ 解压到 `lab/cards/official/{a,b,c,d,alpha,beta,gamma,delta}/`。
>
> 生成的卡、跑分结果（`runs/`）和组装出的包（`build/`）也都没有包含，用 `setup.sh` 和 `tools/` 重新生成。
> 下文中"你上传的""本机"等措辞是当时写给队友本地 AI 助手的，保留原样。

# survey26 本地闭环实验环境

GOSIM 2026 智能体巡天黑客松的本地实验包。里面有官方评测引擎（和平台同一份代码）、84 张带天气的任务卡、
已上线的 v1 智能体、后续改进的核心、大模型环节、一套"改动 → 跑分 → 统计结论 → 把关 → 上传"的工具，以及历史成绩。

## 三步开始

```bash
bash setup.sh                 # 建虚拟环境、装 numpy、造齐 84 张卡、组装候选版本、自检
. .venv/bin/activate
# 然后把 PROMPT.md 交给本地的 AI 编程助手，让它在这个文件夹里工作
```

需要 Python 3.10 以上。`setup.sh` 在 16 核机器上约 2–3 分钟。

## 手动体验一圈

```bash
# 1. 基线：已上线的 v1，在 12 张小卡上（约几分钟）
python3 tools/bench.py --tag v1 --set quick --agent-dir baselines/v1 --env AO_FIXED_EFFORT=1

# 2. 一次实验：新核心对比 v1，自动出结论并记台账
python3 tools/exp.py --tag next --base v1 --set quick --agent-dir agent --env AO_FIXED_EFFORT=1 --note "新核心"
cat results/LEDGER.md

# 3. 上传前把关：v1 核心 + 大模型环节
python3 tools/gate.py --core baselines/v1 --llm --name v1_llm      # 通过后得到 build/v1_llm.zip
```

## 里面有什么

| 路径 | 内容 |
|---|---|
| `PROMPT.md` | 给本地 AI 的完整任务说明：规则、卡、闭环流程、工具、设计、待办、纪律、没验证过的清单 |
| `hackathon-survey26/` | 官方仓库快照：引擎、本地裁判程序、文档、示例智能体 |
| `baselines/v1/` | 已上线的 v1（平台正式 A–D 均分 31148.84），所有对比的基准，不要改 |
| `agent/` | 规则核心的工作副本（v1 之后的四项改动） |
| `llm_stages/` | 三个大模型环节，自动识别 DeepSeek / OpenAI 兼容 / Kimi / Moonshot / Anthropic |
| `build/` | 组装出来的提交包，由工具生成 |
| `cards/official/` | 你上传的官方卡公开输入（A–D、α–δ） |
| `cards/SETS.json` | 84 张卡的配方和分组；卡由 `setup.sh` 现场造出来 |
| `tools/` | `exp.py` 实验、`bench.py` 跑分、`compare.py` 统计、`health.py` 健康、`gate.py` 把关、`assemble.py` 组装、`make_cards.py` 造卡，以及分析工具 |
| `runs/` `results/` | 历史成绩；`results/LEDGER.md` 是自动追加的实验台账，`results/PLATFORM.md` 手工记平台分数 |

## 任务卡

官方卡的 ZIP（包括练习卡 α–δ）只有公开输入，没有天气，所以官方卡本身不能本地打分。
这里的做法是用官方卡的真实目录，配上官方公开的天气模拟器按种子生成的天气、事件和限时请求。

| 组 | 卡集 | 张数 | 用途 |
|---|---|---:|---|
| `quick` | dev、prac_dev | 12 | 小卡，最快的反馈 |
| `tune` | real_dev、var_dev | 20 | A–D 规模，主要调参场 |
| `confirm` | hold、prac_hold、real_hold、var_hold | 32 | 只在准备采纳改动时确认 |
| `test` | real_test、var_test | 16 | 封存，定最终版本前跑一次 |
| — | smoke | 4 | A–D 真实目录的前 8 夜，只查崩溃和超时 |

- `real_*`：A、B、C、D 的真实目录，每张 2 个天气种子。
- `var_*`：站点、季节、赛季长度、目标数、光纤布局、计分零点都随机，模拟隐藏卡 E–H 的变化。
- 想要更多样本：在 `cards/SETS.json` 里加配方（新种子），运行 `python3 tools/make_cards.py`，再 `--record`。

## 这个包的作者验证了什么、没验证什么

作者的机器只有 2 核，所以：

- 验证过：84 张卡全部能造出来并通过官方加载器；工具链在干净目录里从 `setup.sh` 跑通；
  新核心在四张冒烟卡（A、B、C、D 真实目录的前 8 夜）上正常结束；
  "v1 核心 + 模型环节"的把关流程（结果见本文件末尾）。
- 没验证：68 张大卡一张都没有完整跑过；真实大模型从没接过；16 核上各组要跑多久未知。

这些正是要在你的机器上完成的部分，`PROMPT.md` 第 9 节按优先级列好了。

## 许可

官方数据、任务卡和评测程序归主办方所有（CC BY-NC 4.0，见 `hackathon-survey26/LICENSE.md`），使用请引用
GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/)。
`cards/official/` 是比赛平台发布的任务卡公开输入，仅用于本队参赛，不要再分发。

## 作者机器上的自检记录（2 核，10 月 5 日）

干净目录解压后运行 `bash setup.sh`：84 张卡全部造出，校验值与记录一致；候选包组装并通过平台打包代码的检查。

`python3 tools/gate.py --core baselines/v1 --llm --name v1_llm`（15.2 分钟）：

```
[PASS] build: build/v1_llm.zip
[PASS] unit: geometry equals the official engine
[PASS] unit: sky tracker
[PASS] unit: fault verdict path (good, veto, garbage, error)
[PASS] smoke: four short A-D cards finish cleanly
[PASS] same: with the model off the scores equal the core's: A8 -29063.211791 vs -29063.211791, C8 -26974.71166 vs -26974.71166
[PASS] model good: stages called and their answers applied: briefings applied 8, reviews applied 2, real time 150s
[PASS] model garbage: run stays healthy on rules: briefings applied 0, reviews applied 0, real time 149s
[PASS] model error: run stays healthy on rules: briefings applied 0, reviews applied 0, real time 185s
[PASS] model slow: run stays healthy on rules: briefings applied 1, reviews applied 1, real time 211s
GATE PASSED (10 checks)
```

（冒烟卡只有 8 夜，必观测目标来不及做完，所以总分是负数，这是正常的。）

`tools/exp.py` 的完整流程（补跑基线、跑新版本、健康检查、配对比较、记台账）在两张冒烟卡上跑通过。

两张 8 夜短卡上的一个发现（新核心，固定力度），供参考：把光纤边缘余量 `AO_EDGE_MARGIN` 从 0.02° 改到 0.003°，
B8 的目标得分 5204 → 5719，D8 的目标得分 1560 → 1794。详见 `PROMPT.md` 第 9 节 P3。
