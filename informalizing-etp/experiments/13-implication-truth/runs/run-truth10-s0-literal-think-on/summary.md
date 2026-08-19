# Truth-judgment run: seed=0, per_bin=10, bins=2:8, form=literal, reasoning=on

| model | correct | wrong | unparseable | api-error | graded | acc% | true acc% | false acc% | ans-true% | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 63 | 4 | 3 | 0 | 70 | 90.0 | 88.6 | 91.4 | 49.2 | 1514 |
| qwen/qwen3-32b | 64 | 3 | 3 | 0 | 70 | 91.4 | 94.3 | 88.6 | 53.7 | 2309 |
| meta-llama/llama-3.3-70b-instruct | 42 | 28 | 0 | 0 | 70 | 60.0 | 22.9 | 97.1 | 12.9 | 0 |
| openai/gpt-5-mini | 64 | 5 | 1 | 0 | 70 | 91.4 | 85.7 | 97.1 | 44.9 | 1088 |
