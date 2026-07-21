# Benchmark run: seed=0, n=30, form=two-stage, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 16 | 12 | 0 | 12 | 0 | 0 | 40 | 70.0 | 0 | 0 |
| qwen/qwen3-32b | 1 | 7 | 1 | 29 | 2 | 0 | 40 | 22.5 | 32 | 1 |
| meta-llama/llama-3.3-70b-instruct | 15 | 9 | 0 | 16 | 0 | 0 | 40 | 60.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 5 | 12 | 0 | 22 | 1 | 0 | 40 | 42.5 | 0 | 0 |
| openai/gpt-4o-mini | 4 | 4 | 0 | 28 | 4 | 0 | 40 | 20.0 | 0 | 0 |
| google/gemini-2.5-flash | 23 | 4 | 0 | 9 | 4 | 0 | 40 | 67.5 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 14 | 13 | 0 | 13 | 0 | 0 | 40 | 67.5 | 0 | 0 |
| openai/gpt-5-mini | 20 | 3 | 0 | 10 | 7 | 0 | 40 | 57.5 | 0 | 0 |
