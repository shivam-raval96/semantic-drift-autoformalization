# Truth-judgment run: seed=0, per_bin=10, bins=2:8, form=symbolic, reasoning=on

| model | correct | wrong | unparseable | api-error | graded | acc% | true acc% | false acc% | ans-true% | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 64 | 2 | 4 | 0 | 70 | 91.4 | 91.4 | 91.4 | 50.0 | 1750 |
| qwen/qwen3-32b | 64 | 5 | 1 | 0 | 70 | 91.4 | 91.4 | 91.4 | 49.3 | 2147 |
| meta-llama/llama-3.3-70b-instruct | 38 | 32 | 0 | 0 | 70 | 54.3 | 20.0 | 88.6 | 15.7 | 0 |
| openai/gpt-5-mini | 65 | 4 | 1 | 0 | 70 | 92.9 | 88.6 | 97.1 | 46.4 | 1344 |
