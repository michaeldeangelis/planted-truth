# Planted truth

The guardrails catch search. They miss bias.

I gave an automated research system data with no effect in it and asked whether its safeguards noticed. Analyses whose search the system could see passed at most 1% of the time. An analysis that ran the same search and reported only its best result passed 9%. An analysis that adjusted for a variable measured after the exposure — a collider, which manufactures an effect out of nothing — passed 33% of the time with every safeguard switched on, and every individual step of it would survive a methods review.

That gap is the result. Four of the five guardrails in this system are aimed at an analyst who tried many things and reported the best one. None of them is aimed at an analyst who ran one defensible analysis that was wrong.

The tasks are generated, so the answer is known before anything runs. A null is exactly null. A real effect is 0.19 standard deviations, planted. The system under test never sees which is which.

## What stopped what

The strongest guardrail turned out to be re-running the frozen analysis on data with the exposure column shuffled, and comparing the real answer to what the same procedure produces when there is nothing to find. It cut hidden search from 34 passes in 360 to 3, and cost an honest single comparison one detection in eight. Head to head against holding out 30% of units for replication it was better on both axes: 3.3% against 10.8% on the adversary, 25% against 6% detection on a real effect. The held-out split is the expensive guardrail and the weaker one, and it does not bind an analysis that simply re-searches inside whichever half it is handed — 11 passes in 12.

But the same permutation that makes this work on a search is why it fails on bias. Shuffling the exposure destroys the very confounding the analysis is exploiting, so the comparison runs come back unbiased and the real answer looks extreme by contrast. A null guard built on permutation can only measure what permutation disturbs.

## What I got wrong

Three measurements came back cleaner than the truth before one was right. The first reported no false positives at all, because the analyses could not pass anything: the system had started requiring replication on held-out data and my scripts ignored the split. The second named a commit it had not run, because copying a git worktree copies a pointer to the live repository. The third reported 320 failures that did not exist, because I was reading the file the analysis wrote rather than the one the system decided from. Every error flattered the result. A planted effect of 0.8 standard deviations, detected 10 times out of 10, is what told the zeros apart from a broken harness.

Then I wrote the attacks myself, measured the guard at zero, and said so. Two red teams that had not seen me build it found six ways through in an evening. The cheapest was one line comparing the data file's line endings: the comparison runs were written by a CSV writer and the real one was byte-copied, so a script could tell which run it was in and hand back something unremarkable whenever it was being checked. That took the same fabrication from 1 pass in 30 to 30 in 30.

They also broke the check I had written to keep a red team honest. It asked whether a script mentions `read_csv`, whether its answers vary, whether it reproduced, and whether the system's rejection text contained one particular phrase — and it certified that fabrication as a defensible analysis, thirty times out of thirty. It was the same fault the red team found in the system itself, where the required negative control is whatever the analysis says it is and is never recomputed. A self-check that never recomputes anything is decoration. It now re-runs each candidate on identical data under each transformation the system applies, and requires the same number back.

So: every rate here is an upper bound set by how hard the thing was attacked, and I would not trust one that the author of the guardrail also attacked. `ocb/lab/redteam.py` exists to make that someone else's job.

## Reproduce

Nothing here calls a model. The analyses are scripted and the system under test runs locally, so a full table is minutes of CPU and no spend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

Tests that drive the research lab skip unless `OCB_RLAB_PATH` points at a local checkout of it. The planted tasks and the scorer stand alone:

```bash
python run_baselines.py
```

That runs four scripted analysts over the five task types and prints how often each claims an effect that is not there. `always_null` and `always_effect` are the bounds. `textbook` is one unadjusted comparison and sits near the nominal rate. `fishing` reports the best of everything it tried.

## What else is here

`ocb/tasks.py` generates the tasks: an exact null, a null with many outcomes and subgroups to search, a real effect at about 50% power, a variable measured after the exposure that inverts the sign if you adjust for it, and a correction that arrives mid-analysis and reverses the answer. Everything comes from a seed, so fresh seeds give fresh instances. The generators are public and the seeds behind any frozen evaluation are not, which is the only contamination defence that survives publication.

`ocb/score.py` turns structured claims into verdicts against the planted truth. Deterministic, no model in it.

`ocb/lab/` audits a running research lab. `canary.py` plants questions whose answers live outside the lab's reach, `audit.py` reads the lab's own files and flags weak decision rules, fake controls, selection among many comparisons, citations that do not resolve, and results that drifted after they were frozen. `sim.py` drives scripted analysts through the real kernel. `space.py` is the declared space of analysis choices, 432 of them, that the measurements claim to cover.

`attacks/` holds the red teams' candidates, including the ones that still work.

`docs/analysis-plan.md` is append-only and dated. The rules for each experiment were written before it ran, including the ones that killed results I wanted.

## Limits

The system under test is a private research lab, so the numbers about it cannot be reproduced from this repository alone. The harness is general and the tasks are not.

Permutation calibration assumes rows are exchangeable under the null. That holds for the randomised exposure and independent patients here. It fails for repeated measures, family structure and population stratification, and when it fails the empirical null is too narrow, so the guardrail weakens in the permissive direction.

The analyses are scripted, not written by a model. One live question has run end to end. Whether language models behave like any of these scripted analysts is the open question, and this repository does not answer it.

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
