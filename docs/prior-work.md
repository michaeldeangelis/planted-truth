# Prior work and positioning (2026-09-29, revised 2026-10-01)

Source: web scan by a research subagent. The three closest works below were spot-checked on arXiv; other items are unverified secondhand summaries.

## Bottom line

"Agents p-hack / overclaim" is not new. As of Sep 2026:
- Bertran et al. (PNAS, Jul 2026): AI analysts shift conclusions under persona pressure.
- Miao, Pritchard, Zou, "Agentic Garden of Forking Paths" (arXiv 2607.01507, Jul 2026): personas reproduce 72% of the human ideological gap; proposes the m-value. Real data, no planted truth. [verified]
- OverclaimBench (Smyth et al., arXiv 2609.20812, Sep 2026): coding agents misreport file-review completeness; planted defects. Code review, not statistics. [verified]
- Luo, Kasirzadeh, Shah, "Hidden Pitfalls of AI Scientist Systems" (arXiv 2509.08713): leakage, selection bias, metric misuse. Taxonomy from two systems, no planted-truth benchmark. [verified]
- Also: IntegrityBench (multiple choice), K-Bench (rubric, no truth), TruthInsightBench (no planted truth), BAITBENCH / ImpossibleBench / Huang et al. (reward hacking, planted shortcuts).

Not found: an agentic statistical-analysis benchmark with **planted effect truth**, **claim-level scoring**, and an **intervention factorial** reporting false positives and power together. That combination is the contribution. (Probes were dropped 2026-10-01: no open model in use.)

Name collision: OverclaimBench (Smyth et al.) already exists. Rename this project before the repo goes public.

## Positioning

Do not pitch "agents p-hack under pressure". Pitch:

> On tasks where the true effect is known (null, small, superseded, leaked, forking), how often do Claude-based analysis agents claim what is false, which failure modes drive it, and which safeguards reduce it without losing real effects?

Falsifiable version: see the thesis in `design.md`.

## Hooks, ranked by novelty per effort

1. **Planted-truth overclaim taxonomy.** False positive on a null, inflated effect, buried negative, ignored superseding evidence, leakage-driven claim. Plus a self-contradiction flag: the conclusion contradicts the agent's own logged analysis. Behavioral only, cheap.
2. **Forks explored vs. forks reported.** Test whether the false-positive rate scales with specification search on known nulls. This validates or refutes Miao's m-value against ground truth.
3. **Safeguards: false-positive reduction vs. power.** Pre-analysis plan x forced null control x adversarial reviewer. Report fewer false positives and retained detection of small real effects together. Prior evidence says reviewers are gameable (BadScientist, Huang et al. 2026), so a negative for reviewers is still a result.
4. **Interpretability arm (future work; needs open model + GPU budget).** Compare 2x2 cells: null vs. real, crossed with honest vs. overclaiming conclusion.
   - Baselines: diff-of-means and existing sycophancy/lying directions.
   - The probe must beat a transcript-only LLM judge that can see the analysis outputs. Anthropic's Aug 2026 posts (lie detectors failing to generalize; CHIVE: interp tools not beating transcript-only) say this is hard. A clean negative is acceptable.
   - Control for textual leakage (Boxo et al.).
   - "Model knows" needs an elicitation independent of the written report.
5. **Superseding-evidence tasks.** Nothing comparable was found in analysis-agent benchmarks. Links to filemem-bench.

## Reuse, don't rebuild

- Datasets and templates: BLADE (github.com/behavioral-data/blade), P-Bench, SleuthBench (injection method).
- Harness patterns: OverclaimBench (transcript-verified claims), BAITBENCH (planted traps), filemem-bench (own harness).
- Interventions: POPPER (sequential falsification), the frozen mini-paper idea from Anthropic's automated-researcher post.
- Probes (future work only): Apollo deception-detection repo, Liars' Bench, MASK.

## Overlap checks (resolved 2026-09-29): all partial, none major

- **Hypogenic "Taxonomizing Alignment Failures in AI-driven Research"** (blog hypogenic.ai/blog/weekly-entry-260504; repo github.com/Hypogenic-AI/align-failures-taxonomy-db56-claude).
  - AI-agent-generated artifact; idea credited to Chenhao Tan. Not a paper.
  - Negative omission goes from 0% to 27% (GPT-4.1, p=.002) and 40% (GPT-5) under publish pressure. Single-turn, n=30, regex-scored, toy results table.
  - 80% of 20 MLR-Bench papers omit negatives (LLM-judge audit).
  - Use: cite as motivation for a publish-pressure factor. We differ by running real multi-step analysis, planting several truth types, and scoring individual claims.
- **P-Bench / Fisher-R1** (Miao, Mu, Chen, Zou; arXiv 2608.07437, Aug 2026).
  - Agentic R sandbox, 425 real-data tasks with answer keys.
  - Scores the final effect, p-value, and reject decision only. No scoring of written claims, no overclaim metric, no safeguards.
  - Its RL training generator plants null, borderline, and medium effects. Borrow the idea.
- **IntegrityBench** (Tripathi et al.; "arXiv 2608.12345", ID unverified because the abs page says v1 3 Jun 2026).
  - Single-shot misconduct classification under a pressure ladder, not agentic, no effect truth.
  - Cite with the ID marked unverified.
