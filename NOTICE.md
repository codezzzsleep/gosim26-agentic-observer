# 第三方材料说明 / Notice

本仓库中的代码和文档由本队编写，采用 MIT 许可（见 `LICENSE`）。

以下属于 GOSIM 2026 Agentic Observer Hackathon 主办方的材料，按 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（署名、非商业）提供，**没有包含**在本仓库中：

- 评测引擎、本地裁判程序、官方示例智能体（官方仓库 / 示例包）；
- 任务卡（A–D、α–δ、本地卡 L1–L4）及其目录、天气、事件数据；
- 平台页面与文档。

`lab/tools/` 中的部分工具在运行时导入官方仓库的模块（例如用官方的天气模拟器和目录生成器造本地测试卡），
需要使用者自行获取官方仓库，并遵守其许可。`lab/cards/SETS.json` 只记录造卡的配方（种子和参数），不含卡片数据。

`agent/skymath.py` 依据比赛公开文档中的天球几何公式独立实现。

请引用：GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/)。主办方将发布介绍本基准的正式文章，发布后请引用该文章。

比赛规则（第 9 节）：选手自己编写的代码不受上述许可限制。

---

The code and documentation in this repository were written by the team and are released under the MIT License.
The organizers' materials ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), attribution, non-commercial) — the evaluation engine, local runner, example agents, task cards and their
data, and the platform pages — are not included. Some tools under `lab/tools/` import the official repository at run
time; obtain it separately and follow its license. `lab/cards/SETS.json` holds recipes (seeds and parameters) only. Cite GOSIM 2026 Agentic Observer Hackathon
(https://create.gosim.org/survey26/), or the organizers' paper on the benchmark once it is published.
