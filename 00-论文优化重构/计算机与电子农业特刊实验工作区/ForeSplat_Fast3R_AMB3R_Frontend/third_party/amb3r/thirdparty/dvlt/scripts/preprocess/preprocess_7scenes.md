# 7Scenes preprocessing (evaluation)

Microsoft 7-Scenes (RGB-D, 640x480), following Spann3R: SimpleRecon downloads the
data and generates the projected pseudo-GT depth.

## Download + preprocess

Download the seven scenes (chess, fire, heads, office, pumpkin, redkitchen,
stairs) and write the projected pseudo-GT depth (`frame-*.depth.proj.png`) with
SimpleRecon's
[download + preprocess code](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py)
or the repo-root [`7scenes_preprocess.py`](../../../../7scenes_preprocess.py)
(set `src_folder` to the target). Place the result under
`${user.data_root}/test/sevenscenes`.

Layout:

```text
test/sevenscenes/
└── <scene>/
    ├── TrainSplit.txt
    ├── TestSplit.txt
    └── seq-XX/
        ├── frame-000000.color.png
        ├── frame-000000.depth.proj.png   # projected pseudo-GT depth, uint16 mm
        ├── frame-000000.pose.txt          # 4x4 camera-to-world, metres
        └── ...
```

Consumed by
[`test_datasets/sevenscenes.yaml`](../../config/experiments/data/test_datasets/sevenscenes.yaml)
(`depth_source: proj` reads `.depth.proj.png`; `raw` reads the Kinect
`.depth.png`). Intrinsics are fixed (focal 525, principal point at image centre).
