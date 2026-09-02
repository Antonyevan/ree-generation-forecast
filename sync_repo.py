"""Pulls the latest committed data before anything reads it.

Both the personal tracking dashboard (personal_dashboard.py) and the
ree-assistant project call this before reading latest_metrics.json or any
other file here, so neither one ever silently reads stale local data —
the exact problem discovered when compute_latest_metrics.py first ran
against a six-day-old local copy.
"""

import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def sync():
    """Runs `git pull` in this repo. Returns True on success, False otherwise.

    Never raises — a sync failure (e.g. no internet) should degrade to
    'use whatever local data exists', not crash the caller.
    """
    try:
        result = subprocess.run(
            ["git", "pull", "--no-rebase"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[sync_repo] git pull failed: {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    except Exception as exc:
        print(f"[sync_repo] git pull error: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    ok = sync()
    print("Sync succeeded." if ok else "Sync failed — using existing local data.")
