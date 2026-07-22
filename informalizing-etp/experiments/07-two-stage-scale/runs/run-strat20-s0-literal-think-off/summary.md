# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 65 | 62 | 1 | 26 | 6 | 0 | 160 | 80.0 | 0 | 0 |
| qwen/qwen3-32b | 48 | 54 | 0 | 56 | 2 | 0 | 160 | 63.7 | 145 | 1 |
| meta-llama/llama-3.3-70b-instruct | 110 | 1 | 0 | 45 | 4 | 0 | 160 | 69.4 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 39 | 68 | 0 | 43 | 10 | 0 | 160 | 66.9 | 0 | 0 |
| openai/gpt-4o-mini | 71 | 12 | 0 | 73 | 4 | 0 | 160 | 51.9 | 0 | 0 |
| google/gemini-2.5-flash | 107 | 33 | 0 | 14 | 6 | 0 | 160 | 87.5 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 64 | 75 | 0 | 20 | 1 | 0 | 160 | 86.9 | 0 | 0 |
| openai/gpt-5-mini | 96 | 8 | 0 | 37 | 19 | 0 | 160 | 65.0 | 0 | 0 |
| openai/gpt-5.5 | 159 | 0 | 0 | 1 | 0 | 0 | 160 | 99.4 | 0 | 0 |
