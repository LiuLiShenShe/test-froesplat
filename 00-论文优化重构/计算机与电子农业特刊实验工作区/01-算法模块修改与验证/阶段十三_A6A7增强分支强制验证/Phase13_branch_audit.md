# Phase 13 P0 — Production Code Audit

日期：2026-08-31

## A6 — Cross-View Consensus

```
flag:       --use_cross_view_consensus (action=store_true, default=False)
default:    OFF
entry:      main() L2740-2776
trigger:    args.use_cross_view_consensus == True
input:      selected_by_stem (dict[str, ndarray]) — Pass 1 选中掩膜
            gray_by_stem (dict[str, ndarray]) — 灰度图 512×910
            colmap_observations (dict[str, ColmapObservation]) — COLMAP 3D 点
output:     ConsensusResult (per_frame_masks, per_frame_info, geo_support, center_band_mask)
            OR None (insufficient frames < consensus_min_frames)
fallback:   None return → consensus_summary = {"状态": "skipped_insufficient_frames"}
            Per-frame: if frame lacks consensus → uses original selected_by_stem[stem]
evidence:   提示词选择.csv: 共识启用/共识接受/共识回退IoU/共识删除像素比例/共识补回像素比例
            运行日志.json: consensus_accepted
colmap数据: args.colmap_dir → load_colmap_observations() → ColmapObservation.points
            如果 colmap_dir is None → 返回空 dict → geo_support = None
```

## A7 — Memory Propagation

```
flag:       --use_memory_propagation (action=store_true, default=False)
default:    OFF
entry:      main() L2778-2816
trigger:    args.use_memory_propagation == True
input:      image_paths (dict[str, Path])
            seed_stem (str) — 选自 scores_by_stem 最高分帧
            seed_prompt_text (str) — default_prompt_id 对应文本
            seed_box_mask (ndarray) — base_for_memory[seed_stem]
            stems_in_order (list[str]) — 帧顺序
output:     memory_masks (dict[str, ndarray]) — per-frame propagated masks
            memory_info (dict) — 状态/种子帧/传播方向/后端
memory初始化: load_sam3_video_predictor(args) → SAM3 video predictor
memory读取:   predictor.handle_stream_request(propagate_in_video) → per-frame masks
memory写入:   predictor.handle_request(add_prompt) on seed frame → seeds memory state
帧顺序依赖:   stems_in_order 保证时间顺序; bidirectional=True 双向传播
异常处理:     try/except Exception → memory_masks={}, info["状态"]="unavailable:..."
日志字段:     记忆传播.json + 提示词选择.csv: 记忆后端/记忆种子帧/记忆候选采用
候选产生:     Pass 2 L2832-2835: if memory_masks and stem in memory_masks → Candidate("A7记忆")
最终scoring:  Pass 2 L2836-2847: 所有 variants 统一 score_candidate() → max(total_score)
```

## Pass 2 Reprompt Detection

```
entry:      Pass 2 L2942-2954 (inside per-frame loop)
trigger:    args.use_reprompt_detection == True AND reprompt_score() > args.reprompt_threshold
函数:       reprompt_score(prev_mask, curr_mask, prev_image, curr_image, weights)
计算:       iou_drop, area_change, ssim_drop, edge_change → 加权 score
输出:       重提示帧标记.csv (图像/重提示分数/IoU下降/面积变化/SSIM下降/边界变化/是否标记)
注意:       这是时序一致性检测，与 Pass 1 score-gap trigger 完全独立
            Pass 1 needs_reprompt → reprompt_stems (dead variable, never read)
            Pass 2 reprompt_flag → 重提示帧标记.csv (recorded but not consumed downstream)
当前状态:   两者都只记录不消费。Pass 2 reprompt 标记后不会重新推理。
```

## Failure Architecture

```
empty candidate:
  Pass 1 L2709-2712: 检测所有候选为空 → print ⚠ → 继续（不跳过帧）
  select_mask: 空候选 → mask 全 False + needs_reprompt=True
  最终输出: 空 mask 仍然写入 最终掩膜/

A6 unavailable:
  L2741: if args.use_cross_view_consensus → 执行
  L237: if n_frames < consensus_min_frames → return None → skipped
  L2605: load_colmap_observations() → if colmap_dir is None → return {}
  无 try/except 包裹 A6 整体 → 任何异常会向上传播

A7 unavailable:
  L2795-2809: try/except Exception → memory_masks={}, info["状态"]="unavailable:..."
  degrade to per-frame mode (no memory)

CUDA OOM:
  L506-508: torch.cuda.OutOfMemoryError → return {}, info["cuda_oom_fallback"]
  仅在 propagate_memory_masks() 内部捕获

model unavailable:
  L857-872: load_sam3() 无 try/except → 任何异常直接传播到 main()
  L597: if needs_sam3 → load_sam3(args) → 无 fallback

RuntimeError (inference):
  L2804: propagate_memory_masks() 内部异常 → 捕获 → fallback
  其他推理路径无 try/except
```
