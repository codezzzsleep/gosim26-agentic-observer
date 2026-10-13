# Agentic Observer — 巡天观测智能体

GOSIM 2026 Agentic Observer Hackathon（https://create.gosim.org/survey26/）的完整参赛项目，
协议 `participant-agent-protocol-v4`（标准输入/输出上每行一个 JSON 对象）。

本版本是**纯规则版**：所有决策由程序计算得出，不调用大模型，不需要任何 API 密钥。

## 目录

```
agent.py               入口：协议主循环、结果处理、故障上报、指向偏差估计、CPU 节奏控制
planner.py             规划器：一次搜索同时决定指向、光纤分配、曝光时长和观测程序
tracker.py             天空/仪器状态跟踪：在（天气水平 W、仪器效率 eta、是否有故障）上做贝叶斯滤波
hazards.py             公报里的方向性事件：地形遮挡、火箭发射、局部天气
state.py               目标目录、已取得的成绩、可重放的曝光账本（用于数据丢失后的状态恢复）
skymath.py             公开的天球几何公式（恒星时、地平坐标、月亮、大气质量、光纤切平面）
observer.project.json  平台清单（构建时安装 numpy，运行 python3 -u agent.py）
requirements.txt       唯一的第三方依赖：numpy
pack_agent.py          打包成可上传的 ZIP
```

## 它怎么做决策

1. **规划（planner.py）。** 每次决策对所有当前可见的目标、20 种候选曝光时长计算收益，然后在
   三十个左右的锚点周围各试几百个指向偏移，按评测引擎完全相同的几何把目标映射到光纤格子，
   每根光纤取格内收益最大的目标；选 `总收益 − 时间价格 × T` 最大的方案，再在细网格上微调。
   一次决策评估上万个（指向，时长）组合。
2. **赛季计划。** 同一目标只计最好的一次曝光，所以"以后还要重做"的半截曝光基本是浪费。程序估计
   剩余赛季能完成哪些目标、一个光纤秒值多少分，据此区分"现在做完能省下以后时间"的目标和
   "只能顺带拍一点"的目标。必观测目标的奖励按"以后来不及做的风险"计入，随赛季推进逐步加大。
3. **天空与仪器跟踪（tracker.py）。** 天气同时影响完成因子和观测程序档位，仪器效率只影响完成因子。
   程序据此从自己的曝光结果里分离两者，并区分三种效率变化：地震（有公报、逐夜恢复）、
   仪器故障（无公报、持续到上报为止）、圆顶临时关闭（短暂、之后恢复）。
4. **故障上报。** 当"存在未修复故障"的概率持续偏高，且期望收益（修复后的增益 × 概率）大于
   误报代价时才上报；免罚次数用完后门槛更高，并限制付费误报的次数。
5. **指向偏差（Hard mode）。** 用每个已指派目标的命中/未命中结果，在偏差网格上做似然估计，
   通常两次曝光后就能修正。
6. **数据丢失（Hard mode）。** 收到 `state_resync` 后按失效窗口重放自己的曝光账本，恢复每个目标的
   最高分和完成因子。
7. **CPU 节奏。** 用 `time.process_time` 计量每次决策的 CPU 时间，与
   `wallclock.remaining_real_cpu_seconds` 比较；只有明显超出预算时才分级降低搜索力度。

程序只使用协议下发的公开信息（目录、公开评分配置、公报、预报、自己的观测结果），不读取任何卡片文件。

## 本地运行

需要 Python 3.9+ 和 numpy：

```bash
pip install --target .deps -r requirements.txt        # 或者直接 pip install numpy
python3 <官方示例包>/runner/run_local.py --inherit-env --card <官方示例包>/local-cards/L1 \
        --agent "python3 -u agent.py" --agent-cwd .
```

## 打包上传

```bash
python3 pack_agent.py --out ../observer-agent.zip
```

在「参赛」页上传 ZIP，或用命令行：

```bash
survey26 project upload observer-agent.zip --title "observer agent"
survey26 project wait <版本>        # 构建 + 公开场景测试，不占评测次数
survey26 project confirm <版本> --yes
survey26 eval start <版本> --no-model --yes
```

本版本不调用模型，评测时选「本次不提供模型」/ `--no-model` 即可。

## 可调参数

`planner.py` 和 `agent.py` 顶部的常数都可以用 `AO_<名字>` 环境变量覆盖，例如
`AO_KAPPA=0.7`、`AO_FIXED_EFFORT=1`（固定搜索力度，便于复现对比）。

## 许可

本项目自写代码采用 MIT 许可。比赛数据、任务卡与评测程序归主办方所有（CC BY-NC 4.0），
请引用 GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/)。
