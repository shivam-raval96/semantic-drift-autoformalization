# Two-Stage Paired Analysis

**Experiment 07: 160 ETP pairs x 9 models, reasoning off, N = 1,440**

- **Rescued:** Direct Story -> RG was wrong, but Two-Stage was correct.
- **Lost:** Direct Story -> RG was correct, but Two-Stage was wrong.

## Table 1. Overall Paired Transitions

| Direct Story -> RG | Two-Stage | Cases |
|---|---|---:|
| Correct | Correct | 628 |
| Wrong | Correct | **235 rescued** |
| Correct | Wrong | **202 lost** |
| Wrong | Wrong | 375 |
| **Net gain** | **235 - 202** | **+33** |

**Overall accuracy: 57.6% -> 59.9% (+2.3 percentage points)**

## Table 2. Two-Stage Effect by Model

| Model | Rescued | Lost | Net Change |
|---|---:|---:|---:|
| Mistral Small 3.2 | 46 | 28 | **+18** |
| Claude Haiku 4.5 | 41 | 24 | **+17** |
| Llama 3.3 70B | 37 | 20 | **+17** |
| GPT-5.5 | 8 | 1 | **+7** |
| GPT-5 Mini | 28 | 23 | **+5** |
| Qwen3 32B | 23 | 21 | **+2** |
| DeepSeek Chat V3.1 | 12 | 21 | **-9** |
| Gemini 2.5 Flash | 14 | 24 | **-10** |
| GPT-4o Mini | 26 | 40 | **-14** |

## Table 3. Two-Stage Effect by Problem Complexity

| Total Operations | Rescued | Lost | Net Change |
|---|---:|---:|---:|
| Low complexity: 1-3 | 47 | 128 | **-81** |
| High complexity: 4-8 | 188 | 74 | **+114** |
| **All problems** | **235** | **202** | **+33** |

## Table 4. Two-Stage Effect by Exact Operation Count

| Total Operations | Rescued | Lost | Net Change |
|---:|---:|---:|---:|
| 1 | 5 | 68 | **-63** |
| 2 | 19 | 34 | **-15** |
| 3 | 23 | 26 | **-3** |
| 4 | 31 | 23 | **+8** |
| 5 | 33 | 20 | **+13** |
| 6 | 39 | 12 | **+27** |
| 7 | 43 | 11 | **+32** |
| 8 | 42 | 8 | **+34** |

## Table 5. Two-Stage Effect by Complexity, Excluding Vacuous Laws

A pair is excluded when either law contains zero operations.

| Total Operations | Observations | Rescued | Lost | Net Change | Story Accuracy | Two-Stage Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2 | 63 | 12 | 2 | **+10** | 79.4% | 95.2% |
| 3 | 90 | 16 | 2 | **+14** | 81.1% | 96.7% |
| 4 | 72 | 21 | 4 | **+17** | 55.6% | 79.2% |
| 5 | 180 | 33 | 20 | **+13** | 62.8% | 70.0% |
| 6 | 180 | 39 | 12 | **+27** | 43.9% | 58.9% |
| 7 | 180 | 43 | 11 | **+32** | 33.3% | 51.1% |
| 8 | 180 | 42 | 8 | **+34** | 31.7% | 50.6% |
| **Total** | **945** | **206** | **59** | **+147** | **49.9%** | **65.5%** |

There are no non-vacuous one-operation pairs because distributing one
operation across two laws necessarily leaves one law with zero operations.

**Key result:** After excluding vacuous laws, Two-Stage has a positive net
effect at every available complexity level. The negative effect below four
operations in the full dataset is therefore driven by vacuous laws, not by low
complexity itself.
