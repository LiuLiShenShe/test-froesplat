# Git 交付前检查

日期：2026-10-15（Asia/Shanghai）。前置提交为 `694743e7201e1fd79b4d7cff89ad72a4db8a9e33`，分支为 `main`；提交前远端 `origin/main` 也指向该提交。

已用明确阶段路径执行 `git add -- <阶段目录>`。本记录添加前的暂存范围为 343 个新增文件，全部位于 `阶段二十E2_多种子稳定性与前景三维评价/`；没有修改或删除历史文件，也没有阶段之外的暂存/工作区改动。本记录同样只加入该阶段目录。

原始 `git diff --cached --check` 将 Python CSV writer 保存的 CRLF 行尾报告为 trailing whitespace。保留所有 CSV、训练日志与报告原始字节及其已记录 SHA256，不作换行重写。采用允许 CRLF 的检查口径后，退出码为 0、告警数为 0：

```bash
git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --cached --check
```

该设置仅用于单次检查，没有更改仓库 Git 配置或 whitespace 属性。交付目录不含 checkpoint、PLY、float tensor 或超过 100 MiB 的文件；大型资产仍由 `reports/LARGE_ARTIFACT_MANIFEST.csv` 索引。最终科学验收为 G1–G18 全部 PASS、16 项测试通过、341 个唯一 float 渲染的 10,230 个数值比较最大绝对误差为 0。失败启动与首次终结失败证据保留且哈希匹配。

来源：实际 Git 状态与暂存列表、CRLF 兼容检查结果、`reports/FINAL_ACCEPTANCE.json`、`reports/FLOAT_RENDER_METRIC_RECOMPUTATION.json`、`reports/FIGURE_REFRESH_PROGRESS.json`。
