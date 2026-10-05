# Contributor workflow

These instructions apply to every human contributor and AI agent working in this repository.

## Before contributing

1. Identify the contributor you are working for from their explicit instructions. Use that contributor's assigned branch (for example, `tanmay`, `amedeo`, or `damien`). If the contributor or branch is unknown, ask before making changes; do not infer identity from Git credentials or the currently checked-out branch.
2. Inspect the current branch and working tree. Preserve uncommitted changes and other contributors' work. Do not reset, discard, or overwrite work to switch branches.
3. Fetch the latest `main` from the shared remote (`git fetch origin main`) before starting a contribution. Fetching alone does not update a working branch.
4. Switch to the contributor's branch. If it does not exist locally, check for the remote branch first and track it; otherwise create it from `origin/main`. Do not create branches for other contributors.
5. Bring new `origin/main` commits into the contributor's branch before editing. Prefer merging on shared branches to avoid rewriting published history. Resolve conflicts while preserving others' changes. If `origin/main` is already an ancestor, no merge is needed.
6. Read `Goals.md` to understand ownership, today's goals, and recorded progress.

## While working

- Keep contributions within the assigned scope and contributor's branch.
- Track actual time spent when possible. Distinguish measured active work, elapsed automation time, and unrecorded time; do not invent durations or sum overlapping work as team effort.
- Record genuine completion and relevant validation. Drafts, pending reviews, and unverified results must remain clearly labelled.

## Before every push

1. Check `Goals.md` and update it if today's goals, ownership, status, or time taken changed. Every workday has its own date heading and a four-column table: **Goals | Contributor | Status | Time taken**.
2. Use an unchecked status (`☐`) for work that is not started, in progress, or awaiting required review, and a checked status (`☑`) only for completed work. Add a brief status description. Use `Not recorded` for unknown durations and `—` for work not started.
3. Preserve other contributors' entries and previous dates. Add today's section if absent. When carrying work forward, leave the earlier record and add a continuation entry under the new date. Add new actual time only once.
4. Review the diff and run checks appropriate to the change. Ensure unrelated files, secrets, local environments, and temporary outputs are excluded.
5. Verify that the current branch belongs to the contributor, then push to that branch on the shared remote. Never push directly to `main` or another contributor's branch unless the human explicitly requests it.
6. Create or update a pull request from the contributor's branch into `main`; reuse an existing open PR rather than creating a duplicate. Use a draft PR for work awaiting agreement or otherwise not ready to merge. Describe changes, validation, and outstanding items clearly.
7. Do not merge the PR without an explicit request. Report the branch and PR link after publishing.
