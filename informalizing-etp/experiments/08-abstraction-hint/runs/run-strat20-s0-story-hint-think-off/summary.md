# Benchmark run: seed=0, n=30, form=story, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 54 | 26 | 0 | 21 | 4 | 0 | 105 | 76.2 | 0 | 0 |
| qwen/qwen3-32b | 8 | 19 | 3 | 67 | 8 | 0 | 105 | 28.6 | 88 | 1 |
| meta-llama/llama-3.3-70b-instruct | 10 | 32 | 0 | 46 | 17 | 0 | 105 | 40.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 11 | 14 | 1 | 71 | 8 | 0 | 105 | 24.8 | 0 | 0 |
| openai/gpt-4o-mini | 21 | 2 | 0 | 81 | 1 | 0 | 105 | 21.9 | 0 | 0 |
| google/gemini-2.5-flash | 35 | 38 | 4 | 27 | 1 | 0 | 105 | 73.3 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 18 | 29 | 0 | 54 | 4 | 0 | 105 | 44.8 | 0 | 0 |
| openai/gpt-5-mini | 42 | 8 | 2 | 43 | 10 | 0 | 105 | 49.5 | 0 | 0 |
| openai/gpt-5.5 | 71 | 28 | 0 | 6 | 0 | 0 | 105 | 94.3 | 0 | 0 |
