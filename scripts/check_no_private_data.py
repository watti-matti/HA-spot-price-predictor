"""Pre-commit guard against accidentally staging private user data.

Rejects a commit if the staged change-set includes:

- any file under `studies/_private/` other than `README.md`
- any `*.db`, `*.db-shm`, `*.db-wal`, `*.db-journal`
- any file named `household_profile.json` or `household_profile_*.json`
- any `*.csv` under a directory named `_raw`

Wire as a Git pre-commit hook:

    # Windows / Git Bash
    cp scripts/check_no_private_data.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

or via pre-commit framework (see `.pre-commit-config.yaml`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


FORBIDDEN_SUFFIXES = (".db", ".db-shm", ".db-wal", ".db-journal")
FORBIDDEN_BASENAMES_PREFIX = ("household_profile",)


def _staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _is_forbidden(path: str) -> str | None:
    p = PurePosixPath(path.replace("\\", "/"))
    parts = p.parts

    # studies/_private/* except README.md
    if "studies" in parts and "_private" in parts:
        idx = parts.index("_private")
        rest = parts[idx + 1:]
        if rest and rest != ("README.md",):
            return "file inside studies/_private/ (only README.md is allowed)"

    name = p.name.lower()
    for suf in FORBIDDEN_SUFFIXES:
        if name.endswith(suf):
            return f"recorder DB file ({suf})"
    for prefix in FORBIDDEN_BASENAMES_PREFIX:
        if name.startswith(prefix) and name.endswith(".json"):
            return "household profile JSON"
    if name.endswith(".csv") and "_raw" in parts:
        return "raw CSV under a _raw/ directory"
    return None


def main() -> int:
    bad: list[tuple[str, str]] = []
    for path in _staged_files():
        reason = _is_forbidden(path)
        if reason:
            bad.append((path, reason))

    if not bad:
        return 0

    print("\nERROR: commit blocked — private user data is staged:\n",
          file=sys.stderr)
    for path, reason in bad:
        print(f"  {path}  ({reason})", file=sys.stderr)
    print(
        "\nIf this is intentional, run with `git commit --no-verify`,\n"
        "but the privacy contract in docs/household_profile_schema.md\n"
        "says this should not happen. Move the file under\n"
        "studies/_private/ (gitignored) instead.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
