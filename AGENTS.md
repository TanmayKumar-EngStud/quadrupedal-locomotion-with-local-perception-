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

1. Check `Goals.md` and update it if today's goals, ownership, status, or time taken changed. Every workday has its own date heading and a four-column table: **Goals | Contributor | Status | Time taken**. Track substantive project checkpoints, not routine branch creation, tracker maintenance, or other administrative activity.
2. Keep each Status cell to `[ ]` for incomplete work or `[X]` for completed work, with no status prose or attribution. Work awaiting required review remains unchecked. Use `Not recorded` for unknown durations and `—` for work not started.
3. Preserve other contributors' entries and previous dates. Add today's section if absent. When carrying work forward, leave the earlier record and add a continuation entry under the new date. Add new actual time only once.
4. Review the diff and run checks appropriate to the change. Ensure unrelated files, secrets, local environments, and temporary outputs are excluded.
5. Verify that the current branch belongs to the contributor, then push to that branch on the shared remote. Never push directly to `main` or another contributor's branch unless the human explicitly requests it.
6. Stop after pushing to the contributor's branch and report the changes, validation, branch, and commit in the chat. The contributor reviews the changes, requests revisions if needed, validates the result, and creates the pull request themselves.
7. Never create a pull request, including a draft, on your own. A request to edit, commit, or push code is not authorization to create a PR. Do not edit PR metadata, close a PR, or merge a PR unless the human explicitly requests that action. Branch pushes may automatically appear in an already-open PR; this does not authorize further PR actions.
