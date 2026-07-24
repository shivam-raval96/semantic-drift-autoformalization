# Base evals (Phase 3) — eval_v1, no-think, greedy, bf16 via vLLM/Modal

| model | arm | tier | N | correct% | exact | swap | dual | wrong | unparse | unparse% | len-cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | story | normal | 180 | 0.0 | 0 | 0 | 0 | 110 | 70 | 38.9 | 0 |
| Llama-3.2-1B-Instruct | story | hard | 197 | 0.0 | 0 | 0 | 0 | 129 | 68 | 34.5 | 0 |
| Llama-3.2-1B-Instruct | story | extra_hard | 200 | 0.0 | 0 | 0 | 0 | 127 | 73 | 36.5 | 0 |
| Llama-3.2-1B-Instruct | story | order5 | 200 | 0.0 | 0 | 0 | 0 | 126 | 74 | 37.0 | 0 |
| _Llama-3.2-1B-Instruct · story: wall 166.6s, gen 15.3s, NVIDIA A10_ | | | | | | | | | | | |
| Llama-3.2-1B-Instruct | literal | normal | 180 | 0.0 | 0 | 0 | 0 | 177 | 3 | 1.7 | 0 |
| Llama-3.2-1B-Instruct | literal | hard | 197 | 0.0 | 0 | 0 | 0 | 187 | 10 | 5.1 | 0 |
| Llama-3.2-1B-Instruct | literal | extra_hard | 200 | 0.0 | 0 | 0 | 0 | 200 | 0 | 0.0 | 0 |
| Llama-3.2-1B-Instruct | literal | order5 | 200 | 0.0 | 0 | 0 | 0 | 185 | 15 | 7.5 | 0 |
| _Llama-3.2-1B-Instruct · literal: wall 174.0s, gen 15.2s, NVIDIA A10_ | | | | | | | | | | | |
| Llama-3.1-8B-Instruct | story | normal | 180 | 0.0 | 0 | 0 | 0 | 164 | 16 | 8.9 | 1 |
| Llama-3.1-8B-Instruct | story | hard | 197 | 0.0 | 0 | 0 | 0 | 177 | 20 | 10.2 | 5 |
| Llama-3.1-8B-Instruct | story | extra_hard | 200 | 0.5 | 0 | 1 | 0 | 187 | 12 | 6.0 | 3 |
| Llama-3.1-8B-Instruct | story | order5 | 200 | 0.0 | 0 | 0 | 0 | 153 | 47 | 23.5 | 9 |
| _Llama-3.1-8B-Instruct · story: wall 647.3s, gen 404.8s, NVIDIA A10_ | | | | | | | | | | | |
| Llama-3.1-8B-Instruct | literal | normal | 180 | 1.1 | 2 | 0 | 0 | 154 | 24 | 13.3 | 1 |
| Llama-3.1-8B-Instruct | literal | hard | 197 | 0.5 | 1 | 0 | 0 | 166 | 30 | 15.2 | 3 |
| Llama-3.1-8B-Instruct | literal | extra_hard | 200 | 0.0 | 0 | 0 | 0 | 167 | 33 | 16.5 | 4 |
| Llama-3.1-8B-Instruct | literal | order5 | 200 | 0.0 | 0 | 0 | 0 | 110 | 90 | 45.0 | 9 |
| _Llama-3.1-8B-Instruct · literal: wall 570.6s, gen 390.1s, NVIDIA A10_ | | | | | | | | | | | |
