---
name: pre-commit
description: Run the pre-commit checklist before committing — checks branch, tests, code quality, and suggests a conventional commit message
argument-hint: Optionally pass an issue number (e.g. 22) to override the auto-detected issue ref
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(pytest:*), Bash(python:*)
---

# Pre-Commit Checklist

Run this before every commit on the linkedin-agent project.

## Step 1 — Branch check

Run `git branch --show-current`. If the current branch is `main`, **stop immediately** and tell the user:

> You are on `main`. Never commit directly to `main`. Create a feature branch first:
> ```
> git checkout -b feat/<issue-number>-<short-description>
> ```

Branch naming convention:
- `feat/<issue-number>-<short-description>` — new feature
- `fix/<issue-number>-<short-description>` — bug fix
- `docs/<issue-number>-<short-description>` — docs only

Only continue if the branch is not `main`.

## Step 2 — Run all tests

```bash
pytest -v
```

Report the number of passed, failed, and errored tests. If **any tests fail**, stop and tell the user to fix them before committing. Do not proceed.

## Step 3 — Show staged changes

```bash
git diff --staged --stat
```

Summarise which files are staged and what kind of changes they contain.

## Step 4 — Code quality check

Scan the staged diff (`git diff --staged`) and warn the user if any of the following are found:

- `print(` statements (debug output)
- Large blocks of commented-out code
- Hardcoded secrets or API keys (patterns like `sk-`, `ghp_`, `AIza`)
- `pytest.mark.skip` or `.only` in test functions

These are warnings, not blockers — let the user decide whether to proceed.

## Step 5 — Suggest a conventional commit message

Infer the commit message from:
1. The branch name — extract the issue number (e.g. `feat/22-streamlit-gui` → `#22`, type → `feat`)
2. The staged diff — summarise what changed in under 72 characters
3. The explicit argument if passed (overrides the auto-detected issue number)

Format:
```
<type>(<issue-ref>): <short description>
```

Examples:
```
feat(#22): add Streamlit chat GUI
fix(#23): handle duplicate user registration
docs(#27): update README with v2 setup guide
test: add unit tests for auth service
```

Omit the issue ref for cross-cutting changes (`refactor:`, `test:`).

## Step 6 — Confirm

Show the suggested commit message and ask:
> Proceed with this commit message, adjust it, or cancel?

Do not run `git commit` until the user confirms.

## Rules
- Never commit or push directly to `main`
- Do not commit if any tests fail
- Do not commit debug prints or hardcoded secrets
- One logical unit of work per commit — no micro-commits
- Commit messages must follow Conventional Commits format
