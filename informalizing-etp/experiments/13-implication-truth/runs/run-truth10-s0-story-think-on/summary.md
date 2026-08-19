# Truth-judgment run: seed=0, per_bin=10, bins=2:8, form=story, reasoning=on

| model | correct | wrong | unparseable | api-error | graded | acc% | true acc% | false acc% | ans-true% | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 62 | 4 | 4 | 0 | 70 | 88.6 | 82.9 | 94.3 | 47.0 | 2537 |
| qwen/qwen3-32b | 53 | 10 | 7 | 0 | 70 | 75.7 | 62.9 | 88.6 | 41.3 | 3840 |
| meta-llama/llama-3.3-70b-instruct | 36 | 34 | 0 | 0 | 70 | 51.4 | 14.3 | 88.6 | 12.9 | 0 |
| openai/gpt-5-mini | 61 | 8 | 1 | 0 | 70 | 87.1 | 77.1 | 97.1 | 40.6 | 1856 |
