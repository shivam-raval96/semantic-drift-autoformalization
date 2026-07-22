# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 12 | 6 | 0 | 11 | 1 | 0 | 30 | 60.0 | 0 | 0 |
| qwen/qwen3-32b | 1 | 5 | 0 | 22 | 2 | 0 | 30 | 20.0 | 25 | 1 |
| meta-llama/llama-3.3-70b-instruct | 9 | 0 | 0 | 18 | 3 | 0 | 30 | 30.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 1 | 6 | 0 | 21 | 2 | 0 | 30 | 23.3 | 0 | 0 |
| openai/gpt-4o-mini | 2 | 0 | 0 | 24 | 4 | 0 | 30 | 6.7 | 0 | 0 |
| google/gemini-2.5-flash | 21 | 4 | 0 | 4 | 1 | 0 | 30 | 83.3 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 9 | 7 | 0 | 13 | 1 | 0 | 30 | 53.3 | 0 | 0 |
| openai/gpt-5-mini | 7 | 1 | 0 | 12 | 10 | 0 | 30 | 26.7 | 0 | 0 |
