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

**Key result:** Two-Stage hurts on easy problems but helps substantially on complex problems.
