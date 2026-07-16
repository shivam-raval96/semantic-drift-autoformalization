# Benchmark run: seed=0, n=30, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 15 | 2 | 0 | 8 | 5 | 0 | 30 | 56.7 | 0 | 0 |
| qwen/qwen3-32b | 1 | 0 | 0 | 28 | 1 | 0 | 30 | 3.3 | 24 | 1 |
| meta-llama/llama-3.3-70b-instruct | 2 | 3 | 0 | 16 | 9 | 0 | 30 | 16.7 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 2 | 0 | 0 | 22 | 6 | 0 | 30 | 6.7 | 0 | 0 |
| openai/gpt-4o-mini | 0 | 0 | 0 | 26 | 4 | 0 | 30 | 0.0 | 0 | 0 |
| google/gemini-2.5-flash | 13 | 4 | 0 | 12 | 1 | 0 | 30 | 56.7 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 1 | 4 | 0 | 25 | 0 | 0 | 30 | 16.7 | 0 | 0 |
| openai/gpt-5-mini | 5 | 0 | 0 | 19 | 6 | 0 | 30 | 16.7 | 0 | 0 |
