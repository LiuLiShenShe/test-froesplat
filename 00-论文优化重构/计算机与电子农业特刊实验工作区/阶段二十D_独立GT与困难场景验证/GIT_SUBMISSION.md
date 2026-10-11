# Phase20D public submission

Source baseline: `898cec2af3aa121511716c3fb288b00ede4468b3`, clean `main`.
Only this new Phase20D directory is submitted. Original RGB, context, masks, GT and
checkpoints are not submitted or overwritten; no unrelated large cache is included.
Only public CSV/JSON/Markdown/Python/text logs and isolated attributes are staged.

Public artifact commit: `05f3118f` —
`docs(phase20d): audit frozen TEST32 inputs and block validation pending human GT`.
86 new public files, 541767 bytes; all within this new stage. Publication check rejects
symlinks, image/checkpoint binaries, files over 1 MB and API credential strings.
`git diff --cached --check` passes; original artifact/code boundary passes and focused
annotation/metadata regression tests report 60 passed.

Normal `git push origin main` succeeds (exit 0):

```text
To github.com:LiuLiShenShe/test-froesplat.git
   898cec2a..05f3118f  main -> main
```

The byte-captured primary push receipt is `logs/git_push_main.txt`.
This follow-up receipt change adds only publication documentation/logs; no experiments,
predictions, annotations or algorithm changes. No force push, reset, amended history
or new branch is used.

Experiment status remains PARTIAL / USER_ANNOTATION_REQUIRED / BLOCKED regardless
of Git publication success; publication cannot make missing human GT pass.
