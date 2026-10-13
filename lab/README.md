# 本地闭环实验环境

排名只看比赛结束后才运行的隐藏卡，而官方发布的任务卡只有公开输入、没有天气，本地无法打分。
所以我们用官方公开的模拟器按种子造出一批**带天气、可以打分**的本地卡，让每个改动在这些卡上先证明自己，再上平台。

这里的工具默认测试仓库根目录的 [`agent/`](../agent/)，也就是最终提交的版本。

## 准备

以下属于主办方的材料不在本仓库中（CC BY-NC 4.0，见 [`NOTICE.md`](../NOTICE.md)），需要自己获取：

| 需要什么 | 放到哪里 |
|---|---|
| 官方仓库或示例包（评测引擎、本地裁判程序） | `lab/hackathon-survey26/`，或用环境变量 `SURVEY26_REPO` 指向它 |
| 官方任务卡 ZIP 的公开输入（A–D、α–δ） | 解压到 `lab/cards/official/{a,b,c,d,alpha,beta,gamma,delta}/` |

另外需要 Python 3.10 以上。

## 快速开始

```bash
cd lab
bash setup.sh quick smoke        # 建虚拟环境、装 numpy、造 16 张卡、自检；不带参数则造全部 84 张
. .venv/bin/activate
python3 tools/gate.py            # 对 ../agent 做上传前把关（约 20 分钟，2 核）
```

`setup.sh` 会核对造出的每张卡与 `cards/CHECKSUMS.json` 中的记录是否一致，同一种子在任何机器上都得到同一张卡。

## 任务卡

| 组 | 卡集 | 张数 | 用途 |
|---|---|---:|---|
| `quick` | dev、prac_dev | 12 | 小卡，最快的反馈 |
| `tune` | real_dev、var_dev | 20 | A–D 规模，主要的调参场 |
| `confirm` | hold、prac_hold、real_hold、var_hold | 32 | 准备采纳一个改动时才跑 |
| `test` | real_test、var_test | 16 | 封存，定最终版本前只跑一次 |
| — | smoke | 4 | A–D 真实目录的前 8 夜，只查崩溃和超时 |

- `dev` / `hold`：小目录，模拟天气，难度分四档，对应官方本地卡 L1–L4。
- `prac_*`：练习卡 α–δ 的真实输入，配模拟天气。
- `real_*`：A–D 的真实目录、天区、日程和计分参数，配模拟的天气、事件和限时请求，每张卡两个种子。
- `var_*`：站点、季节、赛季长度、目标数、光纤布局、计分零点都随机，范围覆盖 A–D，用来模拟隐藏卡可能的样子。

所有天气和事件都按官方模拟器的规则生成，不是官方卡的真实天气。本地分数只用来比较版本，不预测平台成绩。
要更多样本，在 `cards/SETS.json` 里加配方（新种子），运行 `python3 tools/make_cards.py`，确认无误后加 `--record` 记入校验表。

## 一圈实验

```bash
python3 tools/package.py --out build/base                    # 把当前版本冻结一份作基线
python3 tools/bench.py --tag base --set quick --agent-dir build/base --env AO_FIXED_EFFORT=1

# 改 ../agent 的代码，或者直接用环境变量试参数（planner.py、agent.py 顶部的常数都可用 AO_<名字> 覆盖）
python3 tools/exp.py --tag kappa07 --base base --set quick --env AO_FIXED_EFFORT=1 AO_KAPPA=0.7 --note "时间价格调高"
```

`exp.py` 一条命令完成：补跑基线缺的卡 → 跑新版本 → 健康检查 → 与基线逐卡配对比较 → 在 `results/LEDGER.md` 记一行。
比较的结论是 BETTER / WORSE / NO DIFFERENCE / UNCLEAR / TOO FEW CARDS，依据是对卡重抽样得到的平均差 95% 区间。

- **小卡固定搜索力度**（`AO_FIXED_EFFORT=1`）：决策可以复现，两次运行分数逐位相同。
- **大卡开着节奏控制**（不设 `AO_FIXED_EFFORT`）：和平台一样按 CPU 预算调整力度，单卡运行噪声约 ±1–2%。

我们做实验时守的几条规矩：

1. 只看多张卡的平均和区间，不看单张卡。
2. `quick` 和 `tune` 用来调，`confirm` 只用来确认，`test` 封存到最后。调参时看了 `confirm` 或 `test`，它们就不再是"没见过的卡"。
3. 一次只改一件事，每个改动配一个环境变量开关；默认值只在确认通过后才改。
4. 不针对某张卡的天气、目标、站点或日期写特殊逻辑，智能体只根据运行时收到的信息做决定。
5. `real_*` 和 `var_*` 都要看：只在 `real_*` 上变好的改动，可能只是适配了 A–D 的目录。
6. 稳定优先于分数：任何健康检查失败都先修，改完核心跑一次 `smoke`。

## 工具

| 工具 | 做什么 |
|---|---|
| `setup.sh` | 环境、造卡、自检 |
| `tools/gen_card.py` `tools/make_cards.py` | 造单张卡 / 按 `SETS.json` 造全部卡并核对校验和 |
| `tools/bench.py` | 用官方本地裁判程序在一组卡上跑一个版本，存结果；`--keep` 保留逐步明细 |
| `tools/compare.py` | 两个结果逐卡配对比较，给出统计结论 |
| `tools/exp.py` | 一次完整实验（跑分、健康检查、比较、记台账） |
| `tools/health.py` | 检查一组结果：是否正常结束、有无内部错误、CPU 和真实时间是否接近上限、误报和必观测缺失 |
| `tools/gate.py` | 上传前把关：打包并用平台代码读回、扫描密钥；几何和跟踪器单元测试；四张冒烟卡；假模型的 9 种情况 |
| `tools/package.py` | 复制一个智能体目录并打包、检查 |
| `tools/mock_llm.py` | 本地假模型服务：正常、无原生工具、文本工具、答案在推理内容里、限流、慢、乱码、报错、超时 |
| `tools/check_model.py` | 用智能体真实的提示词把三个模型环节对真实服务各调一次，报告结果和耗时（密钥从环境变量或 `.env` 读，不打印） |
| `tools/test_skymath.py` `tools/test_tracker.py` `tools/test_verdict.py` | 几何与官方引擎逐项对比；天空跟踪器；故障裁决路径 |
| `tools/inproc.py` | 在同一进程里跑智能体，便于调试和 `--profile` 性能分析 |
| `tools/waste.py` | 一次运行的光纤时间去向：有效、过曝、被更好的曝光取代、零分、空光纤 |

生成的卡（`cards/*/`）、运行结果（`runs/`）、打包结果（`build/`）和台账（`results/`）都不进仓库。
