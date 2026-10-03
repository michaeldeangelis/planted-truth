# Which guardrail earns its power cost?

80 cohorts per task type per arm, pooled over 2 projects, same seeds in every
arm. `fishing_hidden` searches every outcome and subgroup and reports only its best result;
`textbook` is one unadjusted comparison. Lab: patched copy of ca9fe80.

| gates | strategy | false pass, null truth | 95% CI | power on 0.19 SD |
|---|---|---|---|---|
| neither | fishing_hidden | 39/240 | 0.121-0.214 | 59/80 |
| neither | textbook | 0/240 | 0.000-0.016 | 21/80 |
| holdout only | fishing_hidden | 26/240 | 0.075-0.154 | 22/80 |
| holdout only | textbook | 0/240 | 0.000-0.016 | 5/80 |
| calibration only | fishing_hidden | 8/240 | 0.017-0.064 | 30/80 |
| calibration only | textbook | 0/240 | 0.000-0.016 | 20/80 |
| both | fishing_hidden | 2/240 | 0.002-0.030 | 11/80 |
| both | textbook | 0/240 | 0.000-0.016 | 4/80 |
