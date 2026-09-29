---
name: update-readme
description: Update README.md based on verified code changes — finds all closed issues from a starting number, checks each PR diff, and adds or updates only what was actually implemented. Prevents hallucination by never documenting a feature without confirming it exists in the diff.
argument-hint: Pass the starting issue number (e.g. 22). All closed issues from that number onward are processed.
---

# Update README

Update `README.md` based on **verified** code changes for all closed issues from a given starting number. Never document a feature without confirming it exists in the diff.

## Step 1 — Get the starting issue number

Use the argument passed (e.g. `22`). If no argument is provided, ask the user:

> Which issue number should I start from?

## Step 2 — Find all closed issues from that number onward

```bash
gh issue list --state closed --json number,title --jq '.[] | select(.number >= <starting-number>) | {number, title}' | sort -t: -k2 -n
```

Collect the full list. Process each issue in ascending order.

## Step 3 — For each issue: get the linked PR and diff

```bash
gh pr list --state merged --search "closes #<issue-number>" --json number,title,mergedAt
```

If a merged PR is found, get the full diff:

```bash
gh pr diff <pr-number>
```

If no PR is found, warn and skip that issue:

> Issue #N is closed but no linked PR was found. Skipping — the feature may not have been implemented via a PR.

## Step 4 — Cross-check issue vs diff

Read the issue's **Acceptance Criteria** section (via `gh issue view <number> --json body`). For each criterion:

- Search the diff for evidence it was implemented
- Mark each criterion as: ✅ Implemented, ⚠️ Partial, or ❌ Not found in diff

Collect results for all issues. Report the full cross-check summary to the user before touching the README:

```
Issue #22 — <title>
  ✅ Criterion A
  ⚠️ Criterion B (partial)
  ❌ Criterion C (not found in diff — will skip)

Issue #23 — <title>
  ✅ Criterion D
  ...
```

**Never add content to the README for criteria marked ❌.**

## Step 5 — Read the current README

Read `README.md` in full. For each piece of content you plan to add or update:

1. Search the README for existing coverage of that feature (by keyword or section heading)
2. If it **already exists** — update the existing section in place, do not add a duplicate
3. If it **does not exist** — add it as a new section in the appropriate place

Do not add duplicate sections. Do not add placeholder text. Every line added must be backed by Step 4.

## Step 6 — Make the changes

Edit `README.md` using the Edit tool. Process all verified criteria across all issues in one pass. Only touch sections relevant to the verified features. Do not rewrite unrelated sections.

For each change, follow this pattern:
- **New feature** → add a new section or bullet under the appropriate heading
- **Updated behaviour** → update the existing line/section in place
- **Removed feature** → remove or strike the outdated line

## Step 7 — Report

Show the user a summary across all processed issues:
1. Which issues were processed vs skipped (no PR)
2. Which acceptance criteria were ✅ documented, ⚠️ partial, or ❌ skipped
3. Which README sections were added vs updated

Ask the user to confirm before finishing:
> README updated. Review the changes above — any corrections needed?

## Rules

- **Never document a feature without verifying it in the diff** — no assumptions, no memory
- **Never duplicate content** — always check if a section already exists before adding
- **Never rewrite unrelated sections** — scope changes to the issues' features only
- **Warn, don't guess** — if a criterion isn't in the diff, report it and skip it
- **Check, then write** — Step 4 must complete before Step 6 begins
- **Report first** — show the full cross-check to the user before making any edits
