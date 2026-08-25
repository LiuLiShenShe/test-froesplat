# Skills 调用与任务协作建议

本文件依据 `Claude-Code-Skills完整分析.md`，把后续工作拆成适合调用的 skill 类型。当前拆解已使用：

- `doc-coauthoring`：把大方案重构为可读、可执行的文档体系。
- `firecrawl-parse`：将本地 `F2DMAS.docx` 解析为可检索 markdown，用于提取论文现有证据。
- `research-ideation`：按研究问题、方法选择、实验验证和风险拆分任务。

## 1. 论文与研究任务

| 任务 | 推荐 skill | 使用时机 |
|---|---|---|
| 重写 Abstract/Introduction/Discussion | `doc-coauthoring` | 需要连续共同打磨文本 |
| 设计研究问题和贡献点 | `research-ideation` | 需要定位创新点和 gap |
| 检查论文完整性 | `paper-self-review` 或 `code-review-excellence` 的审查思路 | 投稿前或版本冻结前 |
| 检查引用真实性 | `citation-verification` | 增加或替换参考文献时 |

## 2. 算法实现任务

| 任务 | 推荐 skill | 备注 |
|---|---|---|
| 实现 M3-M5 代码 | `daily-coding` | 修改代码时使用，遵守单分支可开关约束 |
| 新增 Dataset/Model registry | `architecture-design` | 只有需要 `@register_*` 或 factory 时使用 |
| 调试训练崩溃、指标异常 | `bug-detective` | 先复现再定位，不直接猜修复 |
| 代码审查 | `code-review-excellence` | 重点查 baseline 是否被污染 |

## 3. 文档和网页资料任务

| 任务 | 推荐 skill | 备注 |
|---|---|---|
| 查询库/API 最新用法 | `find-docs` + 项目 AGENTS 中的 `ctx7` 规则 | 必须先 `npx ctx7@latest library` |
| 抓取网页论文或在线资料 | `web-access` 或 Firecrawl 系列 | 用户明确要求联网时使用 |
| 解析本地 DOCX/PDF | `firecrawl-parse` | 解析结果保存到 `.firecrawl/` |

## 4. 后续 agent 协作建议

如果后续显式要求使用子代理并行，可以按以下边界分派：

| Agent | 负责范围 | 写入边界 |
|---|---|---|
| 数据 agent | 样本清单、GT 可用性、资源路径 | `01-数据与资源/` 和数据表 |
| 训练入口 agent | 配置、参数、组合标签、日志保存 | 2DGS 训练脚本与配置模块 |
| M3 agent | mask loss 和 bg opacity loss | loss、render alpha、mask loader |
| M4 agent | pruning score 和剪枝记录 | pruning 模块、metrics |
| M5 agent | edge-aware TSDF 或 mesh correction | meshing 模块 |
| 实验 agent | 实验矩阵脚本和表格汇总 | `03-实验设计/` 与实验脚本 |
| 写作 agent | 正文、图表 caption、结果叙事 | `04-论文写作与图表/` |

协作原则：

- 不同 agent 的写入文件必须尽量不重叠。
- 所有实现 agent 都要先读 `00-项目总览/02-开发约束与组合规则.md`。
- 每个 agent 返回时必须列出修改文件、运行命令、验证结果和未完成风险。

