# Benchmark run: seed=0, n=30, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 25 | 5 | 0 | 9 | 1 | 0 | 40 | 75.0 | 0 | 0 |
| qwen/qwen3-32b | 8 | 1 | 2 | 26 | 3 | 0 | 40 | 27.5 | 22 | 1 |
| meta-llama/llama-3.3-70b-instruct | 11 | 13 | 0 | 16 | 0 | 0 | 40 | 60.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 11 | 4 | 0 | 17 | 8 | 0 | 40 | 37.5 | 0 | 0 |
| openai/gpt-4o-mini | 15 | 0 | 0 | 24 | 1 | 0 | 40 | 37.5 | 0 | 0 |
| google/gemini-2.5-flash | 22 | 10 | 1 | 7 | 0 | 0 | 40 | 82.5 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 10 | 12 | 1 | 17 | 0 | 0 | 40 | 57.5 | 0 | 0 |
| openai/gpt-5-mini | 18 | 0 | 1 | 18 | 3 | 0 | 40 | 47.5 | 0 | 0 |
