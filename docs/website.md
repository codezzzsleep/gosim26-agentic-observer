# 项目展示页

线上地址：<https://codezzzsleep.github.io/gosim26-agentic-observer/>

`site/` 是独立的静态站点，使用 HTML、CSS、JavaScript 和内联 SVG，无构建步骤、第三方前端依赖或模型请求。首屏、页面标题与分享摘要明确展示“GOSIM 2026 决赛三等奖”，奖项等级与最终第 6 名分别呈现。页面围绕 cloudbrain 的获奖作品、项目交付成果、核心实现和验证结果展开；比赛名次作为成果的外部验证来源，巡天坐标与电蓝配色作为视觉语言。支持亮色与暗色，默认跟随系统外观，手动选择通过 localStorage 保存；星场、坐标图和光纤图随主题重绘。

页面顺序为：项目首屏 → 交付成果 → 核心实现 → 规划示意 → 实现验证 → 验证工具 → 开源交付。84 张卡的分层与迭代流程默认收进可展开的细节。验证区域使用项目主题变量，不能固定为亮色。

## 本地预览

在仓库根目录运行：

```bash
python3 -m http.server 8000 --directory site
```

打开 <http://localhost:8000/>。页面使用相对资源路径，可部署在 GitHub Pages 的项目子目录下。

## 内容维护

- `site/index.html`：项目介绍、成绩、模块说明、实验方法与来源链接。
- `site/style.css`：桌面、平板与手机样式。
- `site/app.js`：固定星场的观测示意、模块切换、手机导航与复制命令。
- `site/favicon.svg`：站点图标。
- `site/fonts/`：本地托管的 Space Grotesk 与 IBM Plex Mono，附 SIL OFL 许可。

更新样式、脚本或图标后，同步更新 `index.html` 中相应资源 URL 的 `?v=` 版本标记，以避免旧资源缓存。

成绩以 [官方最终榜单](https://create.gosim.org/survey26/platform/leaderboard/final-hidden) 为来源。实验方法与本地验证记录来自仓库 README、`agent/README.md` 和 `lab/README.md`。

交互观测使用确定性生成的 4×4 示例星场，每格四个目标；根据示例天气、曝光时长与既有成绩，比较程序和每格的最佳分配。自动选择时长采用固定的示例时间价格。它用于解释“一根光纤一个目标”和“只保留最好的一次曝光”，没有调用实际 Python 规划器，也不代表比赛回放或成绩预测。

## 发布

GitHub Pages 从 `gh-pages` 分支的根目录发布。该分支只包含 `site/` 的内容，`.nojekyll` 禁用 Jekyll；智能体、实验工具、环境配置和其他源码不进入站点部署目录。

先提交修改，再从仓库根目录运行：

```bash
bash ops/publish_pages.sh --dry-run
bash ops/publish_pages.sh
```

脚本使用 `git subtree split --prefix=site HEAD` 导出已提交的站点，再普通推送到 `gh-pages`。它拒绝尚未提交的站点修改，不强制覆盖远端分支。推送后由 GitHub Pages 的分支部署流程发布，可在仓库的 Actions 或 Deployments 页面查看状态。

首次设置在 GitHub 仓库 **Settings → Pages** 中选择 **Deploy from a branch → gh-pages → / (root)**。已经配置过的仓库不需要重复设置。

站点修改需要再次运行发布脚本；推送 `main` 本身不会更新线上页面。
