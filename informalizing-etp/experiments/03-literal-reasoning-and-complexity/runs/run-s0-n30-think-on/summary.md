# Benchmark run: seed=0, n=30, form=literal, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 21 | 9 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 1783 |
| qwen/qwen3-32b | 24 | 3 | 0 | 2 | 1 | 0 | 30 | 90.0 | 30 | 920 |
| meta-llama/llama-3.3-70b-instruct | 14 | 6 | 0 | 8 | 2 | 0 | 30 | 66.7 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 15 | 3 | 0 | 10 | 2 | 0 | 30 | 60.0 | 0 | 0 |
| openai/gpt-4o-mini | 9 | 0 | 0 | 15 | 6 | 0 | 30 | 30.0 | 0 | 0 |
| google/gemini-2.5-flash | 26 | 0 | 0 | 3 | 1 | 0 | 30 | 86.7 | 29 | 2750 |
| anthropic/claude-haiku-4.5 | 27 | 0 | 0 | 3 | 0 | 0 | 30 | 90.0 | 30 | 813 |
| openai/gpt-5-mini | 17 | 6 | 0 | 7 | 0 | 0 | 30 | 76.7 | 30 | 2592 |
