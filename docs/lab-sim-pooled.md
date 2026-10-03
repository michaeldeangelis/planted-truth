# Lab kernel vs scripted researcher strategies (no LLM)

Lab: `ca9fe80` [pinned snapshot] at `/private/tmp/ocb-efc8b1c0/lab-pinned`. Re-run this table whenever the lab moves.

120 independent cohorts per task type across 3 project(s) (NULL, FORK, LEAK = null truth; SMALL = -0.19 SD). Verdicts come from the real rlab kernel; audit findings from `ocb/lab/audit.py`.

| strategy | false pass, all null-truth | NULL | FORK | LEAK | effect recorded significant though null | SMALL detected | SMALL inflated or wrong sign | audit fail/warn |
|---|---|---|---|---|---|---|---|---|
| textbook | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 27/360 | 8/120 | 0/120 | 0/3 |
| adjust_all | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 128/360 | 32/120 | 0/120 | 0/3 |
| fishing | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 86/360 | 5/120 | 1/120 | 6/9 |
| fishing_matched | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 86/360 | 7/120 | 3/120 | 10/11 |
| fishing_hidden | 34/360 (0.0684-0.1291) | 3/120 | 30/120 | 1/120 | 86/360 | 35/120 | 8/120 | 42/72 |
| declared | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 86/360 | 5/120 | 1/120 | 1/8 |
| bonferroni | 0/360 (0.0-0.0106) | 0/120 | 0/120 | 0/120 | 5/360 | 0/120 | 0/120 | 0/3 |
