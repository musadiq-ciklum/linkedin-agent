---
name: create-pr
description: Create a draft GitHub PR for the current branch — title taken from the first commit, body auto-generated from the issue and diff
argument-hint: Optionally pass an issue number (e.g. 22) to override auto-detection
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(git log:*), Bash(git push:*), Bash(gh pr:*), Bash(gh issue:*), Bash(gh auth:*), Bash(gh repo:*)
---

# Create Pull Request

## Step 1 — Branch check

Run `git branch --show-current`. If the branch is `main`, **stop immediately**:

> You are on `main`. Switch to a feature branch before creating a PR.

Only continue if the branch is not `main`.

## Step 2 — Extract issue number and type

Parse the current branch name using this priority:
1. Explicit argument passed to the skill (e.g. `22`)
2. Branch name pattern: `feat/22-streamlit-gui` → issue `22`, type `feat`

If no issue number can be detected, ask the user to provide one.

## Step 3 — Get the PR title from the first commit

Run:
```bash
git log main..HEAD --oneline --reverse
```

Take the **first commit message** on the branch — this is the primary commit that describes the feature. Use it verbatim as the PR title. Do not rephrase it.

Example: if the first commit is `feat(#22): add Streamlit chat GUI`, the PR title is exactly `feat(#22): add Streamlit chat GUI`.

If there are multiple commits, list them all so the user can see what will be in the PR. Do not concatenate them — only the first one becomes the PR title.

## Step 4 — Fetch the GitHub issue title

Run:
```bash
gh issue view <issue-number> --json title,body --jq '{title: .title, body: .body}'
```

Use the issue title to cross-check the PR title makes sense. If the issue cannot be found, warn the user but continue.

## Step 5 — Build the PR body

Generate the PR body with these sections:

```markdown
## Summary
Closes #<issue-number>

<2–3 sentences describing what the PR does and why, based on the diff and issue title>

## Commits
<bullet list of all commits on this branch, in order — use `git log main..HEAD --oneline --reverse`>

## Changes
<file-by-file bullet list from `git diff main...HEAD --stat`>

## Testing
- [ ] All tests pass locally (`pytest -v`)
- [ ] Manually tested the feature end-to-end
- [ ] No debug prints or commented-out code in staged files

## Checklist
- [ ] Commit messages follow Conventional Commits format
- [ ] Unit tests added or updated for changed behaviour
- [ ] `CLAUDE.md` updated if architecture changed
```

## Step 6 — Push the branch

```bash
git push origin <current-branch>
```

If the branch is already up to date with remote, skip silently.

## Step 7 — Create the draft PR

```bash
gh pr create \
  --draft \
  --title "<PR title from Step 3>" \
  --body "<body from Step 5>" \
  --base main \
  --head <current-branch>
```

Always create as **draft**. The user converts it to ready with `gh pr ready <number>` when done.

## Step 8 — Confirm

Print the PR URL and title. Remind the user:
> PR created as draft. Run `gh pr ready <number>` when it is ready for review.

## Rules
- PR title must be the exact text of the first commit on the branch — never rephrase it
- Always target `main` as the base branch
- Always create as draft
- Never push or create a PR from `main`
- `Closes #<issue-number>` must appear in the PR body so GitHub auto-links the issue
