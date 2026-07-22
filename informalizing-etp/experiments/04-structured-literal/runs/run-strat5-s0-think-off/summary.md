# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 18 | 18 | 0 | 3 | 1 | 0 | 40 | 90.0 | 0 | 0 |
| qwen/qwen3-32b | 9 | 12 | 0 | 18 | 1 | 0 | 40 | 52.5 | 32 | 1 |
| meta-llama/llama-3.3-70b-instruct | 32 | 0 | 0 | 8 | 0 | 0 | 40 | 80.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 11 | 13 | 0 | 12 | 4 | 0 | 40 | 60.0 | 0 | 0 |
| openai/gpt-4o-mini | 22 | 1 | 0 | 12 | 5 | 0 | 40 | 57.5 | 0 | 0 |
| google/gemini-2.5-flash | 28 | 6 | 0 | 5 | 1 | 0 | 40 | 85.0 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 18 | 15 | 0 | 7 | 0 | 0 | 40 | 82.5 | 0 | 0 |
| openai/gpt-5-mini | 24 | 4 | 0 | 6 | 6 | 0 | 40 | 70.0 | 0 | 0 |
