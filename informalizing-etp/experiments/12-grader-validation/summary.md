# Experiment 12: grader validation (V-A specification curve, V-B rescue grading)

Re-graded 10320 committed responses from 6 run dirs, offline.
Replay fidelity: 10320/10320 verdicts identical (0 mismatches).

## V-A: does the headline survive the grader's degrees of freedom?

| variant | correct | wrong (silent) | unparseable (loud) | silent:loud |
|---|---|---|---|---|
| replay | 5380 | 3364 | 1576 | 2.1 |
| strict_convention | 3878 | 4866 | 1576 | 3.1 |
| no_dual | 5342 | 3402 | 1576 | 2.2 |
| first_line | 5323 | 3368 | 1629 | 2.1 |
| plain_lines | 5380 | 3364 | 1576 | 2.1 |

Model-ranking stability (Spearman rho vs replay):

- strict_convention: rho = 0.867
- no_dual: rho = 1.000
- first_line: rho = 1.000
- plain_lines: rho = 1.000

Silent:loud ratio by ops bin (bin = (ops_total-1)//2, capped 7):

| variant | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | bin6 | bin7 |
|---|---|---|---|---|---|---|---|---|
| replay | 219/10 | 365/43 | 493/71 | 793/121 | 319/61 | 288/154 | 248/223 | 639/893 |
| strict_convention | 648/10 | 803/43 | 839/71 | 1033/121 | 335/61 | 307/154 | 250/223 | 651/893 |
| no_dual | 248/10 | 371/43 | 495/71 | 794/121 | 319/61 | 288/154 | 248/223 | 639/893 |
| first_line | 219/10 | 365/44 | 493/72 | 793/128 | 319/69 | 288/160 | 248/231 | 643/915 |
| plain_lines | 219/10 | 365/43 | 493/71 | 793/121 | 319/61 | 288/154 | 248/223 | 639/893 |

## V-B: anatomy of silent failures (default-wrong answers)

Tags are diagnostic, deliberately-illegal rescues; a row can carry several tags; 'structural' = no near-miss explains it.

| tag | count |
|---|---|
| structural | 1493 |
| e_wrong_only | 1091 |
| f_wrong_only | 756 |
| onesided_dual | 24 |

By form:

- literal: structural 397/906 tags (44%)
- story: structural 637/1317 tags (48%)
- two-stage: structural 459/1141 tags (40%)
