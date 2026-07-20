# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 21 | 1 | 0 | 6 | 2 | 0 | 30 | 73.3 | 0 | 0 |
| qwen/qwen3-32b | 26 | 1 | 0 | 3 | 0 | 0 | 30 | 90.0 | 21 | 1 |
| meta-llama/llama-3.3-70b-instruct | 18 | 0 | 0 | 11 | 1 | 0 | 30 | 60.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 15 | 0 | 0 | 14 | 1 | 0 | 30 | 50.0 | 0 | 0 |
| openai/gpt-4o-mini | 20 | 0 | 0 | 9 | 1 | 0 | 30 | 66.7 | 0 | 0 |
| google/gemini-2.5-flash | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 25 | 0 | 0 | 1 | 4 | 0 | 30 | 83.3 | 0 | 0 |
| openai/gpt-5-mini | 17 | 1 | 0 | 4 | 8 | 0 | 30 | 60.0 | 0 | 0 |
