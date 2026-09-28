#!/usr/bin/env bash
# Recreate the gitignored symlinks declared in external-paths.json. Idempotent.
#
#   scripts/link-external.sh            create what is missing
#   scripts/link-external.sh --dry-run  report the plan, change nothing
#
# Plain python3 on purpose (not `uv run`): a fresh checkout runs this before it has
# a virtualenv. Existing links are never rewritten and real directories never deleted.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" - "${repo_root}" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

repo = Path(sys.argv[1])
dry_run = "--dry-run" in sys.argv[2:]
spec = json.loads((repo / "external-paths.json").read_text())
root = Path(os.environ.get(spec["artifact_root_env"], spec["default_artifact_root"]))

failed = False
for entry in spec["entries"]:
    link = repo / entry["path"]
    target = root / entry["target"]
    if link.is_symlink():
        state = "ok" if link.exists() else "BROKEN (target missing)"
        print(f"[keep] {entry['path']} -> {os.readlink(link)} ({state})")
        failed |= entry["required"] and not link.exists()
        continue
    if link.exists():
        print(f"[keep] {entry['path']} is a real path; not replaced")
        continue
    if not target.exists():
        if entry.get("create_target"):
            print(f"[mkdir] {target}")
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[MISSING] {entry['path']}: target {target} does not exist. {entry.get('fix', '')}")
            failed |= entry["required"]
            continue
    print(f"[link] {entry['path']} -> {target}")
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
sys.exit(1 if failed else 0)
PY
