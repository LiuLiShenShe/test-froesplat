# Push Instructions

This local repository is ready to push from the owner's GitHub account.

Current local path:

```bash
/data/fj/F2DMAS/ForeSplat_real_release
```

## 1. Log in with the owner account

```bash
gh auth logout
gh auth login
gh auth status
```

Make sure `gh auth status` shows the GitHub account that should own the new `ForeSplat` repository.

## 2. Create the remote repository

Option A: create with GitHub CLI:

```bash
cd /data/fj/F2DMAS/ForeSplat_real_release
gh repo create ForeSplat --public --source=. --remote=origin --push
```

Option B: create the repository on github.com first, then push:

```bash
cd /data/fj/F2DMAS/ForeSplat_real_release
git remote add origin https://github.com/<YOUR_ACCOUNT>/ForeSplat.git
git push -u origin main
```

If an `origin` remote already exists, replace it:

```bash
git remote remove origin
git remote add origin https://github.com/<YOUR_ACCOUNT>/ForeSplat.git
git push -u origin main
```

## 3. Verify after upload

```bash
git status --short --branch
gh repo view <YOUR_ACCOUNT>/ForeSplat --web
```

The repository should contain real non-full ForeSplat data under `examples/`, not synthetic demo data.
