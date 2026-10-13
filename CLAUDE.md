# CLAUDE.md

Portfolio project: a production-grade SMS spam detection API, built to show an end-to-end
path from model to production (target roles: MLOps / AI Engineer).

## Stack
Python, FastAPI, scikit-learn, pandas, joblib, pytest, ruff, Docker, GitHub Actions,
MLflow (optional `mlops` extra), Prometheus/Grafana, Evidently (optional `monitoring` extra),
AWS (planned).

## Working rules
- The owner studies 1 hour/day: keep each session to one small, finished, tested task.
- Talk to the owner in informal Brazilian Portuguese and explain the *why* of decisions.
- Everything that goes to GitHub is in English: code, comments, commits, README, issues, PRs.
- Commits follow Conventional Commits (feat:, fix:, docs:, chore:, test:, ci:).
- The owner works on Windows: give cmd/PowerShell commands only (no `$VAR`, no trailing `\`).
- Ask for confirmation before anything that creates or changes something outside this
  machine (push, create repo, open/close issues, GitHub or AWS settings).
- Show what will be in a commit before committing.
- Never put secrets in files (AWS keys, tokens, .env). Use `.env.example` with no values.
- If a command fails: show the error, explain it in one line, then try something else.
- Work on short branches (feat/..., fix/...) and open a PR instead of committing to main.
  Reference the issue in the PR ("Closes #N").
- `pytest` and `ruff check .` must pass before committing.
