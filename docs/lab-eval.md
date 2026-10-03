# The research lab as the overclaim-bench test bed

The lab is Mike's autonomous research system (`~/.dev/research-lab`, branch `v1/rlab`; projects in `~/.dev/rlab-projects/`). LLM workers triage, search the literature, write a preregistration, implement it as a script, and review. The kernel runs the script itself, applies the preregistered rule, and writes the card and registry entry. This repo measures whether its verdicts can be trusted.

The lab moves fast: on 2026-10-01 it went through roughly ten commits in an evening, shipping sandbox confinement, cross-question read denial, parallelism, holdout confirmation, BH and LORD++ false-discovery control, a rule stage frozen before any data access, and a kernel-owned results file. Two findings from this bench were adopted into it the same evening. Every measurement here therefore names the lab commit it was taken against, and expires when the lab moves.

## Instruments (`ocb/lab/`, CLI `ocb_lab.py`)

| piece | what it does |
|---|---|
| `canary.py` | Builds a lab project of planted-truth questions (NULL, FORK, SMALL, LEAK cohorts; one settled and two invented-study literature questions). Opaque ids, neutral project name, truth held out of worker reach, cohorts disguised and placed inside each question's own folder. |
| `audit.py` | Reads lab files only. Truth-free checks for any project, plus planted-truth scoring for the canary. Never raises on malformed lab output. |
| `sim.py` | Seven scripted researcher strategies driven through the REAL kernel. No LLM, no cost. |
| `space.py` | The declared analysis space the bench claims to cover: 432 cells, the product of six analysis choices, with a script generator for any cell. |
| `bridge.py` | `snapshot_lab()` freezes the lab for measurement and stamps the commit into the snapshot. |

## Baseline: lab `ca9fe80`, 120 cohorts per task type, pooled over 3 projects

Zero unfinished questions, zero fail findings for the honest strategies.

| strategy | false pass, null truth | power on 0.19 SD |
|---|---|---|
| textbook | 0/360 (ceiling 1.06%) | 8/120 |
| adjust_all | 0/360 | 32/120 |
| fishing | 0/360 | 5/120 |
| fishing_matched | 0/360 | 7/120 |
| **fishing_hidden** | **34/360 = 9.4%** [6.8-12.9] | 35/120 |
| declared | 0/360 | 5/120 |
| bonferroni | 0/360 | 0/120 |

Two things follow. Every strategy whose search the kernel can see is held at or below a 1.06% ceiling — the guardrails work. A script that runs the same search and reports only its best result gets 9.4%, and 25% on the many-outcome task, because the multiplicity guard can only correct over comparisons that are reported. And honest power is low: 8 detections in 120 of a real 0.19 SD effect, though covariate adjustment recovers that to 32/120, so part of the power problem is analysis quality rather than gate strength.

## Patches handed to the lab

`docs/lab-patches/`. Not applied here: the lab repo belongs to another session.

**0001 — citations, multiplicity, direction.** Cited ids must resolve and titles must match; Bonferroni over max(declared comparisons, distinct effects reported), applied against the preregistered threshold on the log scale for ratio metrics; kills significant in the opposite direction say so. **Adopted by the lab on 2026-10-01** (`rlab/cite.py` and the `rules.py` guard are in the lab now).

**0002 — permutation calibration and analysis regimes.** Applies cleanly to the live lab; 96 tests pass there (88 existing plus 8 new).

*Calibration* answers the 9.4% hole. The kernel re-runs the frozen script on data whose declared exposure column it has permuted. Whatever search the script performs it performs again with no signal present, so the spread of the reported statistic is an empirical null for the procedure as a whole — the kernel never needs to see inside the script. It fails closed (no declared exposure, too few affordable runs, or too few runs producing a statistic all leave a pass provisional) and is off by default, because it costs one analysis run per permutation and needs at least 19 for an empirical p of 0.05.

*Regimes* answer the power cost. A regime is a named, pre-committed profile chosen by the rule stage before it may see any data, and it may only tighten: alpha capped at the project's and at 0.05, `null_runs` floored at the project's, confirmation not waivable. One exception: a regime committing to a single pre-specified comparison may use the whole sample instead of a discovery/holdout split, because calibration tests the same thing and tests it better — and the kernel counts what is actually reported and withdraws the exemption otherwise. A screening regime's passes are labelled leads on the card, never reported as findings. An unknown regime is refused rather than ignored, so a typo cannot silently buy a looser bar.

The design rule behind both: **agents select among pre-committed regimes; they never set a number.** During development an agent-set `alpha` of 1.0 disabled the multiplicity guard completely, which is the whole argument in one line.

## Method, learned the hard way

Three high-n runs were discarded on 2026-10-01 before one was valid. Each failure made the lab look *better*, which is why they were nearly reported.

1. **Always run a positive control.** A run returned 0/230 false passes and a 1.6% ceiling. The lab had started gating passes on holdout replication and my scripted analyses ignored the split, so nothing could pass at all and the zero was arithmetic. A 0.8 SD planted effect detected 10/10 is what licenses reading a zero as evidence.
2. **Report power beside the false-pass rate, always.** The two failures above were caught only because the power column was also zero. A gate that drives false passes to zero can drive power to zero and read as success.
3. **Pin by stamping, not by copying.** `cp -R` of a git worktree copies `.git` as a pointer to the live repository, so git inside the copy reports whatever the live lab moved on to. One table claimed a commit it had not run.
4. **Snapshot outside `~/.claude`.** The lab's sandbox denies reading `.claude/jobs`, which is this job's scratch directory, so a snapshot there cannot execute: 16 of 83 tests failed and it looked like the lab was broken. Verify a snapshot passes the lab's own suite before measuring with it.
5. **Measure in the regime the lab is built for.** One project with 400+ questions leaves 25-42% of questions unscored, varying run to run, because the per-stage write-boundary check grows with the project and fails closed. At 40 cohorts per type nothing blocks, so high n comes from pooling independent projects.
6. **Audit what the kernel decided from.** The audit read the worker-writable `run/results.json` and hashed one file, while the kernel had moved to a kernel-owned `kernel/results.json` and a hash of the whole code tree. That produced 320 false "fail" findings per run — the bench overclaiming in the opposite direction.

## Limits

Simulation uses scripted strategies, not LLM behaviour; one live question has been run end to end. The canary is one semi-synthetic cohort family with clean nulls, while the lab's real questions are metascience, genetics and protein design. Coverage of the analysis space is now declared (432 cells) but the ceiling over it has not yet been measured. What a certificate may claim, and what it may not, is fixed in `docs/analysis-plan.md`.
