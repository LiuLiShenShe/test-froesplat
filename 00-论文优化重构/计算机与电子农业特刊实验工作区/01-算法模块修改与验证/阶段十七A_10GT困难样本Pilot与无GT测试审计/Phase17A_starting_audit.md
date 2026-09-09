# Phase 17A — Starting-State Audit

**Generated at (UTC)**: 2026-09-09T04:35:51Z
**Phase root**: `阶段十七A_10GT困难样本Pilot与无GT测试审计`
**Purpose**: read-only verification of pre-execution state for the budgeted 10-GT pilot + unlabeled TEST behavioral audit.

## 1. Repository State

| Item | Value |
|---|---|
| branch | `main` |
| HEAD | `a304af7d122be06e8a084c86f01f1ec829acac83` |
| working tree | `?? 阶段十七A_10GT困难样本Pilot与无GT测试审计/` |

```
a304af7d Add 03-GT-区分_challenge: DouBanLv2 + BaiZhang challenge samples (20 files, 5.9MB)
1103f6f9 Phase 16: TEST 32帧脚本密度选帧, 全清单58帧(26DEV+32TEST)冻结, 测试100/100
474bfa9a Phase 16: 用户手动选帧26帧(BaiZhang/DouBanLv2), 标注10帧, GT QA通过, 测试100/100
587f4f1d Phase 16 修正: 选帧改为多株植物密度(green%×peaks), 60帧, 全部为多株共存场景
611cfa92 Phase 16: 硬案例GT挑战集构建与切分冻结(54帧, 7DEV/8TEST, SHA256锁定)
```

## 2. Phase 16 Manifest Integrity (re-derivation)

| Manifest | frozen SHA256 | re-derived SHA256 | match |
|---|---|---|---|
| full (58 rows) | `792a6724a2d7deaabaa562243dbcbd9f959e65690ee3ec6faa1ab1767a55f807` | `792a6724a2d7deaabaa562243dbcbd9f959e65690ee3ec6faa1ab1767a55f807` | ✅ |
| DEV (26 rows) | `063b33b9d7559eb0400528978993a63cbe0146e44e40f4c5d38fc0dfa7ba3cc6` | `063b33b9d7559eb0400528978993a63cbe0146e44e40f4c5d38fc0dfa7ba3cc6` | ✅ |
| TEST (32 rows) | `7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157` | `7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157` | ✅ |

**TEST SHA256 (immutable for Phase 17A)**: `7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157`
**Matches plan expectation**: ✅

## 3. Challenge GT Masks (Track A targets)

| GT mask | exists | size |
|---|---|---|
| BaiZhang/0033 | ✅ | 43747 B |
| BaiZhang/0034 | ✅ | 42137 B |
| BaiZhang/0035 | ✅ | 41020 B |
| BaiZhang/0036 | ✅ | 41061 B |
| BaiZhang/0037 | ✅ | 37836 B |
| DouBanLv2/0037 | ✅ | 25001 B |
| DouBanLv2/0038 | ✅ | 24614 B |
| DouBanLv2/0039 | ✅ | 24147 B |
| DouBanLv2/0040 | ✅ | 23884 B |
| DouBanLv2/0041 | ✅ | 23723 B |

**Total GT masks**: 10 (expected 10)
**QA formal_p6_valid=True for all 10**: ✅

## 4. Locked TEST Frames (Track B inputs)

| Check | Value |
|---|---|
| TEST rows in frozen manifest | 32 |
| on-disk frames present | 32/32 |
| missing | none |

## 5. COLMAP Availability (Track A samples)

| BaiZhang | COLMAP sparse/0 ✅ | raw frames: 250 |
| DouBanLv2 | COLMAP sparse/0 ✅ | raw frames: 250 |

## 6. Pipeline & Weights

| Resource | path | exists |
|---|---|---|
| pipeline script | `/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py` | ✅ |
| SAM3 checkpoint | `/data/fj/F2DMAS/sam3/sam3.pt` | ✅ |
| python | `/home/test/biosoft/enter/envs/sam3/bin/python` | ✅ |

## 7. GPU Snapshot

```
index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%]
0, NVIDIA RTX A6000, 49140 MiB, 299 MiB, 0 %
1, NVIDIA RTX A6000, 49140 MiB, 11 MiB, 0 %
```

## 8. Audit Conclusion

**Starting-state gate**: PASS — proceed

---

Relevant constraints carried into Phase 17A:
- The 10 GT frames are **PILOT_GT10** (DEV material), not a held-out confirmatory test.
- The 32 locked TEST frames have **no GT**; Track B audits execution/behavior only (no accuracy).
- No pseudo-GT is created; no TEST frame is scored against GT.
- **No automatic git push.**
