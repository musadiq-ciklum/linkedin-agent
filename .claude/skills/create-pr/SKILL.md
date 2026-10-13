---
name: create-pr
description: Create a draft GitHub PR for the current branch — title taken from the first commit, body auto-generated from the issue and diff
argument-hint: Optionally pass an issue number (e.g. 22) to override auto-detection
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(git log:*), Bash(git fetch:*), Bash(git merge:*), Bash(git push:*), Bash(gh pr:*), Bash(gh issue:*), Bash(gh auth:*), Bash(gh repo:*), Bash(pytest:*)
---

# Create Pull Request

## Step 1 — Branch check

Run `git branch --show-current`. If the branch is `main`, **stop immediately**:

> You are on `main`. Switch to a feature branch before creating a PR.

Only continue if the branch is not `main`.

## Step 2 — Merge main into current branch

Fetch and merge the latest `main` to ensure the branch is up-to-date before opening the PR:

```bash
git fetch origin main
git merge origin/main
```

If the merge produces conflicts, **stop immediately**:

> Merge conflict detected. Resolve conflicts with `git mergetool` or manually, then re-run `/create-pr`.

If the merge is already up-to-date, continue silently.

## Step 3 — Extract issue number and type

Parse the current branch name using this priority:
1. Explicit argument passed to the skill (e.g. `22`)
2. Branch name pattern: `feat/22-streamlit-gui` → issue `22`, type `feat`

If no issue number can be detected, ask the user to provide one.

## Step 4 — Get the PR title from the first commit

Run:
```bash
git log main..HEAD --oneline --reverse
```

Take the **first commit message** on the branch — this is the primary commit that describes the feature. Use it verbatim as the PR title. Do not rephrase it.

Example: if the first commit is `feat(#22): add Streamlit chat GUI`, the PR title is exactly `feat(#22): add Streamlit chat GUI`.

If there are multiple commits, list them all so the user can see what will be in the PR. Do not concatenate them — only the first one becomes the PR title.

## Step 5 — Fetch the GitHub issue title

Run:
```bash
gh issue view <issue-number> --json title,body --jq '{title: .title, body: .body}'
```

Use the issue title to cross-check the PR title makes sense. If the issue cannot be found, warn the user but continue.

## Step 6 — Pre-check checklist boxes

Run these checks now and record each result. Use the outcome to set `[x]` or `[ ]` in the PR body.

**Tests passing:**
```bash
pytest -v --tb=no -q 2>&1 | tail -5
```
→ Tick `All tests pass locally` if exit code is 0.

**Debug prints / commented-out code:**
```bash
git diff main...HEAD -- '*.py' | grep -E '^\+.*(print\(|#.*TODO|#.*FIXME|pdb\.set_trace|breakpoint\(\))' | grep -v '^\+\+\+'
```
→ Tick `No debug prints or commented-out code` if output is empty.

**Conventional Commits format:**
```bash
git log main..HEAD --format="%s"
```
→ Tick `Commit messages follow Conventional Commits format` if every line matches `^(feat|fix|docs|test|refactor|chore|style|perf|ci|build|revert)(\(.*\))?!?: .+`.

**Unit tests added or updated:**
```bash
git diff main...HEAD --name-only | grep -E '^tests/'
```
→ Tick `Unit tests added or updated` if any test file appears.

**`CLAUDE.md` updated:**
```bash
git diff main...HEAD --name-only | grep 'CLAUDE.md'
```
→ Tick `CLAUDE.md updated if architecture changed` only if `CLAUDE.md` appears in the diff AND the PR touches architecture-level files (new modules, changed factories, new endpoints). Leave unchecked otherwise.

## Step 7 — Build the PR body

Use the check results from Step 6 to fill in `[x]` or `[ ]` for each item:

```markdown
## Summary
Closes #<issue-number>

<2–3 sentences describing what the PR does and why, based on the diff and issue title>

## Commits
<bullet list of all commits on this branch, in order>

## Changes
<file-by-file bullet list from `git diff main...HEAD --stat`>

## Testing
- [x/\ ] All tests pass locally (`pytest -v`)
- [ ] Manually tested the feature end-to-end
- [x/\ ] No debug prints or commented-out code in staged files

## Checklist
- [x/\ ] Commit messages follow Conventional Commits format
- [x/\ ] Unit tests added or updated for changed behaviour
- [x/\ ] `CLAUDE.md` updated if architecture changed
```

`Manually tested the feature end-to-end` is always left unchecked — only the user can confirm this.

## Step 8 — Push the branch

```bash
git push origin <current-branch>
```

If the branch is already up to date with remote, skip silently.

## Step 9 — Create the draft PR

```bash
gh pr create \
  --draft \
  --title "<PR title from Step 4>" \
  --body "<body from Step 7>" \
  --base main \
  --head <current-branch>
```

Always create as **draft**. The user converts it to ready with `gh pr ready <number>` when done.

## Step 10 — Confirm

Print the PR URL and title. Remind the user:
> PR created as draft. Run `gh pr ready <number>` when it is ready for review.

## Rules
- PR title must be the exact text of the first commit on the branch — never rephrase it
- Always target `main` as the base branch
- Always create as draft
- Never push or create a PR from `main`
- Always merge `origin/main` into the branch before pushing — stop if there are conflicts
- `Closes #<issue-number>` must appear in the PR body so GitHub auto-links the issue
- Pre-tick every checklist item that can be programmatically verified; leave `Manually tested` unchecked always
