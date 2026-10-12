# Reproduce without overwriting history

Prerequisites: original32JSON/RGB,71originalmasks, strictmetadata andall existingfrozenlocalarchives. Versions are inlogs/RUNTIME_ENVIRONMENT.json. Sourcechecks are mandatory; missingfiles must not be guessed.

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_固定轮廓直接评价'
export PYTHONDONTWRITEBYTECODE=1
python3 -m pytest "$STAGE/tests" -q -p no:cacheprovider
python3 "$STAGE/scripts/verify_delivery.py" --output "$STAGE/logs/new_verification.json"
python3 "$STAGE/scripts/rasterize_contours.py" --output "$STAGE/reproduction/new_rasterization"
```

Use a genuinelyNEWemptyoutput directory forrasterreproduction;the scriptrefusesoverwrite. Comparedecodedpixelhashes and original/roundedmappingrecords, nottimestamp-dependentfiles. run_frozen.py refusesexistingrunroots. Exactoriginalproductioncommands aretracked in logs/smoke_*_execution.json andlogs/full_*_execution.json. Replaythosecommands withnewoutput_dir andoriginalfrozeninput/candidate/metadata paths; do NOT runthesameoutputcommand overexistinghistory. RunfourCPU smokes beforebothfullruns, thenverifyrawidentity/fullscoredecomposition/configs beforemetrics. Metricreproduction uses thesavedactualmasks andthe unchangedPhase20Bmetricfunction; deliverytests recomputeall32bothrasterizers andconfirmallsummaryrows. NoSAM3inferenceisneeded.

Theignoredfullrunartifacts are indexed bylogs/full_*_artifacts.csv; localderivedGT64masks/mappingfiles byRASTERIZED_MASK_MANIFEST.csv. No externalmodels/data are silentlydownloaded. TheoriginalhumanGate is neitherrequiredfor thisseparateevaluationnorredefinedby it.

## Exact frozen production replay into a fresh directory

The following reuses the archived production commands, changing only the output
directory. It executes the four required smoke jobs before the two full jobs and
does not call SAM3 inference. Run on the host with the archived input paths intact.

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

stage = Path('00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_固定轮廓直接评价').resolve()
replay = stage / 'reproduction/frozen_cli_replay'
replay.mkdir(parents=True, exist_ok=False)
jobs = ['smoke_base_CaoMei2_0146', 'smoke_r1_CaoMei2_0146',
        'smoke_base_WanNianQing1_0227', 'smoke_r1_WanNianQing1_0227',
        'full_base', 'full_r1']
for job in jobs:
    record = json.loads((stage / 'logs' / f'{job}_execution.json').read_text())
    command = list(record['command'])
    position = command.index('--output_dir') + 1
    command[position] = str(replay / job)
    with (replay / f'{job}_stdout.txt').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
PY
```

Before comparing replay masks, confirm identical candidate IDs, source hashes,
score decomposition, selected mask decoded pixels and all parameters other than
the replay output path. PNG file hashes can depend on encoder versions; the
archived environment fixes those versions and manifests also record pixel hashes.
