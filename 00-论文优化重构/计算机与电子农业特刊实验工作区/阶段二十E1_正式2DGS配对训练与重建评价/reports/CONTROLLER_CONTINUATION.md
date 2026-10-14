# Controller continuation, not optimizer resume

The long-lived matrix process returned exit 143 after HongZhang A completed.
Its exact cause is not established; no claim of an optimizer crash is made.
HongZhang A has a complete iteration-30000 runtime, final PLY and exported
checkpoint, exit code 0, with untouched logs and SHA256. No training was active
when inspected. There was no HongZhang C output yet.

The remaining matrix is continued with `--start-at HongZhang`: completed A is
recognized from its original process record, the last required C run starts from
iteration 1 with the same frozen seed/config, and both HongZhang models undergo
the full frozen-camera evaluation. Earlier closed-loop scenes are not retrained
or retuned. This control-plane continuation is not an interrupted optimizer
checkpoint resume and contributes no partial-model statistics.
