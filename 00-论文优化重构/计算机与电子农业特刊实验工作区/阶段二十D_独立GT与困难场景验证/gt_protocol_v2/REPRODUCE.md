# GT V2 可复现命令与恢复说明

所有命令从仓库根目录执行，CPU即可。当前基准 `6092e2ea`，原人工归档来自
`4d8dcd52`。这是协议/格式/审批准备，不调用 SAM3 推理、R0/R1 评价或 Phase20E。

## 1. 原始资料与当前输出

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_独立GT与困难场景验证'
PKG="$STAGE/gt_protocol_v2"
```

- 原始人工 JSON：`annotation_package/rgb/<frame_id>.json`；canonical annotations 为
  指向它们的符号链接，不是新绘制GT。`completion/GT32_SOURCE_ARCHIVE.csv` 固定其SHA256。
- 原始私有字节备份：`completion/GT32_LOCAL_BACKUP_INDEX.csv` 指向
  `completion/local_gt_backup/<source_sha256>/<frame_id>.json`；32份备份已核验。
- 当前未审批转换尝试：`converted_gt/run_01/`；0个派生JSON，32行阻断索引。
  根目录 `CONVERTED_GT_MANIFEST.csv` 是该真实运行索引的归档副本，不是完成标记。
- 实际 RGB 路径及SHA：冻结输入manifest和 `SOURCE_IDENTITIES.csv`。原候选71个仍由
  Phase20D原输入校验器核验身份、分数、metadata和掩膜哈希。
- 审查图：本地 `REVIEW_INDEX.html`，107张整帧/逐shape图片；
  `review_overlays/intersection_details_v2/INTERSECTION_REVIEW.html`，21张局部自交诊断。
  图像未纳入公开Git，源文件和图件SHA在两个索引CSV中保存，可按下列命令重建。

任何源资料缺失时仅从已验证的真实备份/历史资料恢复到**新的**私有目录；不得用模型
造替代源。恢复后重新验证SHA再注册路径，不自动覆盖现有文件。

## 2. 新建客观审计与真实RGB核查图

为了不覆盖任何人类补录，每次使用一个全新的目录。以下 `replay_01` 必须不存在：

```bash
REPLAY="$PKG/runs/replay_01"
python "$PKG/scripts/prepare_v2_review.py" --output-dir "$REPLAY"
python "$PKG/scripts/render_intersection_details.py" \
  --records-dir "$REPLAY" --output "$REPLAY/review_overlays/intersection_details"
```

审核图只缩放显示真实RGB、原始折线/端点及尚未批准的闭合建议边，不填GT。判定
完整轮廓、明显缺口、实际可见性和困难类型仍只能来自实际人类。自交检测也检查
相邻边回折重叠和零长度边；不删除重复点、不修改坐标。

## 3. 已完成人工工作的补录

以 `HUMAN_REVIEW_REQUIRED_ITEMS.csv` / `REMAINING_HUMAN_CONFIRMATIONS.md` 为简化清单。
具体字段见 `HUMAN_INPUT_GUIDE.md`。真实记录含姓名时用私有 records-dir 保管，不把
身份对应表上传。将已有事实转录到3张V2表，不改变原32份JSON和历史D类决策。
共享真实审核人/时间可登记在批量表中；不替代逐帧依据或75个实际轮廓判断。

当前17个shape的坐标存在自交/回折，需要仅核查这些局部异常；签字不能让有自交的
polygon通过几何Gate。若真有必要人工改点，请另立人工修订版本及可追溯流程；本
转换器不会这样做，也不把它混入当前不改点的确定性转换。

## 4. 人类确认后才转换；独立最终复核后再Gate

```bash
RECORDS="$PKG"  # 或已有真实审核记录的私有目录
CONVERT="$PKG/converted_gt/run_02"  # 必须是新的空目录
python "$PKG/scripts/convert_confirmed_gt.py" \
  --records-dir "$RECORDS" --output "$CONVERT"
```

未确认/非法轮廓仅列阻断，不写派生GT；已确认的有效完整轮廓仅改shape_type，保持
原坐标及顺序。得到真正转换文件后，由实际独立人类确认结果，真实逐帧
`annotation_sha256` 绑定运行manifest中的最终SHA；批量记录如使用也须绑定同一SHA。
脚本不会填审批、人物、时间或视觉依据。

```bash
python "$PKG/scripts/validate_gt_v2.py" \
  --records-dir "$RECORDS" --converted-manifest "$CONVERT/CONVERTED_GT_MANIFEST.csv" \
  --output "$CONVERT/GT_GATE_V2_RESULT.json" --report "$CONVERT/GT32_GATE_V2_REPORT.md"
```

退出0表示全32帧V2就绪，退出2表示需真人记录/几何处理。现有实际执行为退出2，
`evaluation_allowed=false`。即使将来就绪，本脚本也不会执行准确率或宣布方法锁定。

可将新的一次实际准备结果归档到新的输出目录（不覆盖当前报告）：

```bash
ARCHIVE="$PKG/runs/archive_02"
mkdir -p "$ARCHIVE"
python "$PKG/scripts/archive_preflight_material.py" --records-dir "$RECORDS" \
  --conversion-dir "$CONVERT" --intersection-dir "$REPLAY/review_overlays/intersection_details" \
  --output "$ARCHIVE"
```

## 5. CPU测试与保护检查

```bash
python -m pytest "$PKG/tests" "$STAGE/tests" "$STAGE/completion/tests" \
  '00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_FIX_候选元数据与精简细化验证/tests' -q
python -m compileall -q "$PKG/scripts" "$PKG/tests"
git diff --check
git diff --name-only 6092e2ea
```

现有140项测试通过（V2新增57项），包含独立合成人工记录的32帧PASS/阻断端到端
案例、源哈希/派生点篡改拒绝、可见性条件、禁止无确认闭合和派生、批量复核范围、
旧Gate及冻结metadata回归，包含等值plant派生的独立人类来源ledger追溯。测试中的
human字段仅合成fixture，绝不写入真实审核表。
环境现有 pytest_asyncio 弃用告警不影响本阶段测试，未修改无关设置。

## 6. 外部资料与视觉QA边界

绘制接口查阅 Context7 获取的官方 Pillow ImageDraw 文档（`/python-pillow/pillow`）。
联网检索API返回HTTP403，未以搜索摘要代替本地实验来源。Agnes仅作一次显示排版
描述（见 `logs/DISPLAY_LAYOUT_QA.json`），输出未经真人核实、可能误读；不采纳其中
标签/场景语义作为证据，不填写任何GT、可见性、困难类别或审批。
