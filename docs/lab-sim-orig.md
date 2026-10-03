# Lab kernel vs scripted researcher strategies (no LLM)

10 independent cohorts per task type (NULL, FORK, LEAK = null truth; SMALL = -0.19 SD). Verdicts come from the real rlab kernel; audit findings from `ocb/lab/audit.py`.

| strategy | false pass, all null-truth | NULL | FORK | LEAK | effect recorded significant though null | SMALL detected | SMALL inflated or wrong sign | audit fail/warn |
|---|---|---|---|---|---|---|---|---|
| textbook | 1/30 (0.006-0.167) | 1/10 | 0/10 | 0/10 | 2/30 | 4/10 | 0/10 | 3/3 |
| adjust_all | 1/30 (0.006-0.167) | 1/10 | 0/10 | 0/10 | 11/30 | 7/10 | 0/10 | 3/3 |
| fishing | 6/30 (0.095-0.373) | 2/10 | 3/10 | 1/10 | 7/30 | 8/10 | 1/10 | 23/28 |
| fishing_matched | 4/30 (0.053-0.297) | 2/10 | 1/10 | 1/10 | 7/30 | 7/10 | 1/10 | 18/23 |
| fishing_hidden | 6/30 (0.095-0.373) | 2/10 | 3/10 | 1/10 | 7/30 | 8/10 | 1/10 | 9/17 |
| declared | 6/30 (0.095-0.373) | 2/10 | 3/10 | 1/10 | 7/30 | 8/10 | 1/10 | 9/17 |
| bonferroni | 0/30 (0.0-0.114) | 0/10 | 0/10 | 0/10 | 0/30 | 3/10 | 0/10 | 5/9 |
