#!/usr/bin/env bash
# Assemble the public tree. Everything about the private lab stays behind: the patches describe
# its internals, and the hiring notes and CLAUDE.md are working files rather than results.
#
#   scripts/export_public.sh /tmp/planted-truth
#
# Then, from the target directory:
#   git init -b main && git add -A && git commit && git remote add origin ... && git push -u origin main
set -euo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dst="${1:?usage: export_public.sh <target directory>}"

rm -rf "$dst"
mkdir -p "$dst/docs"

# code, tasks, scorer, audit layer, attack harness
cp -R "$src/ocb" "$dst/ocb"
cp -R "$src/tests" "$dst/tests"
cp -R "$src/attacks" "$dst/attacks"
cp "$src/ocb_lab.py" "$src/run_pilot.py" "$src/run_baselines.py" "$dst/"
cp "$src/README.md" "$src/CITATION.cff" "$src/LICENSE" "$src/requirements.txt" "$src/.gitignore" "$dst/"
mkdir -p "$dst/scripts" && cp "$src/scripts/export_public.sh" "$dst/scripts/"

# results and the preregistration; the hiring notes are not a result
for d in analysis-plan design prior-work lab-eval lab-calibration lab-ablation \
         lab-sim-orig lab-sim-patched lab-sim-pooled; do
    cp "$src/docs/$d.md" "$dst/docs/$d.md"
done

find "$dst" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$dst" -name '*.pyc' -delete 2>/dev/null || true

echo "exported to $dst"
echo "excluded: docs/anthropic-hiring-2026.md, docs/lab-patches/, CLAUDE.md, runs/"
