# ChangShouHua2 repaired COLMAP source for M2M3 pilot

This experiment-only source was created because the original final-locked `images/0084.jpg` and `images/0085.jpg` are not readable by PIL.

Only the `images` directory is copied into this prepared source. The unreadable two JPGs were regenerated from same-resolution RGBA frames in the original `images_rgba` directory. COLMAP structure directories such as `sparse`, `distorted`, `stereo`, `input`, and `images_rgba` are symbolic links to the original source.

The original source data are not modified.
