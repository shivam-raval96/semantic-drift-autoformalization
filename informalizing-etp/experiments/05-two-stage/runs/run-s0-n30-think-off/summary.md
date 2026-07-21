# Benchmark run: seed=0, n=30, form=two-stage, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 14 | 7 | 0 | 8 | 1 | 0 | 30 | 70.0 | 0 | 0 |
| qwen/qwen3-32b | 3 | 2 | 0 | 23 | 2 | 0 | 30 | 16.7 | 26 | 1 |
| meta-llama/llama-3.3-70b-instruct | 7 | 4 | 0 | 19 | 0 | 0 | 30 | 36.7 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 2 | 3 | 0 | 21 | 4 | 0 | 30 | 16.7 | 0 | 0 |
| openai/gpt-4o-mini | 2 | 0 | 0 | 24 | 4 | 0 | 30 | 6.7 | 0 | 0 |
| google/gemini-2.5-flash | 16 | 1 | 0 | 11 | 2 | 0 | 30 | 56.7 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 9 | 6 | 0 | 14 | 1 | 0 | 30 | 50.0 | 0 | 0 |
| openai/gpt-5-mini | 10 | 0 | 0 | 13 | 7 | 0 | 30 | 33.3 | 0 | 0 |
