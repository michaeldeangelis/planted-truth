# Does permutation calibration close the hidden-selection hole?

The kernel re-runs the frozen script on data whose declared exposure column it has permuted. Whatever
search the script performs it performs again with no signal present, so the spread of the reported
statistic is an empirical null for the procedure as a whole.

120 cohorts per task type, pooled over 3 projects. Lab: patched copy of ca9fe80.

| strategy | calibration | false pass, null truth | 95% CI | power on 0.19 SD |
|---|---|---|---|---|
| fishing_hidden | off | 34/360 | 0.068-0.129 | 35/120 |
| fishing_hidden | 19 runs | 3/360 | 0.003-0.024 | 15/120 |
| textbook | off | 0/360 | 0.000-0.011 | 8/120 |
| textbook | 19 runs | 0/360 | 0.000-0.011 | 7/120 |
