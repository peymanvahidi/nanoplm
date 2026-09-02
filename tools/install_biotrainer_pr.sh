#!/usr/bin/env bash
# Install biotrainer + biotrainer-core from PR sacdallago/biotrainer#192
# (head repo peymanvahidi/biotrainer, branch fix/v2-migration-findings). The
# pbc_zeroshot_contact eval framework needs the fixes on that PR; released
# biotrainer==2.0.0 crashes there with:
#   TypeError: Got unsupported ScalarType BFloat16
#
# Usage: tools/install_biotrainer_pr.sh [ref] [checkout_dir]
#   ref           Branch, tag, or commit SHA to install.
#                 Default: fix/v2-migration-findings. Pass a commit SHA for a
#                 reproducible install.
#   checkout_dir  Where to clone/checkout biotrainer.
#                 Default: <repo>/scripts/biotrainer-pr192, which falls under
#                 .gitignore's blanket "scripts/" entry. Not
#                 agent_space/biotrainer-pr192 - that checkout is in use for
#                 other work.
#
# Rollback to the released version:
#   .venv/bin/pip3 install --force-reinstall --no-deps biotrainer==2.0.0 biotrainer-core==2.0.0
set -euo pipefail

REF="${1:-fix/v2-migration-findings}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_DIR="${2:-$REPO_ROOT/scripts/biotrainer-pr192}"
PIP="$REPO_ROOT/.venv/bin/pip3"
PY="$REPO_ROOT/.venv/bin/python3"
REMOTE_URL="https://github.com/peymanvahidi/biotrainer.git"

log() { echo ">> $*"; }
err() { echo "ERROR: $*" >&2; }

[ -x "$PIP" ] || { err "$PIP not found. Create the venv first, see README.md."; exit 1; }
[ -x "$PY" ] || { err "$PY not found. Create the venv first, see README.md."; exit 1; }

if [ -d "$CHECKOUT_DIR" ]; then
  log "checkout dir exists: $CHECKOUT_DIR"
else
  log "cloning $REMOTE_URL into $CHECKOUT_DIR"
  git clone "$REMOTE_URL" "$CHECKOUT_DIR"
fi

log "fetching and checking out $REF"
git -C "$CHECKOUT_DIR" fetch origin "$REF"
git -C "$CHECKOUT_DIR" checkout --detach FETCH_HEAD

RESOLVED_SHA="$(git -C "$CHECKOUT_DIR" rev-parse HEAD)"
log "resolved ref '$REF' -> commit $RESOLVED_SHA"

# biotrainer-core is a uv-workspace sub-package inside this same repo
# (biotrainer-core/), installed separately and BEFORE biotrainer: biotrainer
# depends on it, so installing biotrainer first would let pip pull core from
# PyPI instead of picking up this checkout.
log "installing biotrainer-core from $CHECKOUT_DIR/biotrainer-core"
"$PIP" install -e "$CHECKOUT_DIR/biotrainer-core" --no-deps

# --no-deps on both installs, always: resolving biotrainer's dependency
# closure re-resolves shared pins and clobbers nanoPLM's MoE kernels
# (quack-kernels must stay ==0.4.1 for dion, nvidia-cutlass-dsl ==4.4.2 for
# gram-newton-schulz). See issues/03-eval-deps-clobber-moe-kernels.md.
log "installing biotrainer from $CHECKOUT_DIR"
"$PIP" install -e "$CHECKOUT_DIR" --no-deps

log "verifying import in a fresh interpreter"
"$PY" - "$CHECKOUT_DIR" <<'PY'
import os
import sys

checkout_dir = os.path.realpath(sys.argv[1])

import biotrainer
import biotrainer_core

ok = True
for name, mod in (("biotrainer", biotrainer), ("biotrainer_core", biotrainer_core)):
    path = os.path.realpath(mod.__file__)
    print(f"{name}.__file__ = {path}")
    if os.path.commonpath([path, checkout_dir]) != checkout_dir:
        print(
            f"ERROR: {name} resolved outside {checkout_dir}; "
            "a PyPI install may be shadowing the PR checkout.",
            file=sys.stderr,
        )
        ok = False

sys.exit(0 if ok else 1)
PY

log "done: ref=$REF commit=$RESOLVED_SHA checkout=$CHECKOUT_DIR"
