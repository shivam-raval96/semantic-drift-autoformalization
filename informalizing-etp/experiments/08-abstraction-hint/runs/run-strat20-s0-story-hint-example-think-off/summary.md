# Benchmark run: seed=0, n=30, form=story, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 33 | 37 | 0 | 29 | 6 | 0 | 105 | 66.7 | 0 | 0 |
| qwen/qwen3-32b | 6 | 16 | 3 | 79 | 1 | 0 | 105 | 23.8 | 105 | 1 |
| meta-llama/llama-3.3-70b-instruct | 10 | 30 | 0 | 57 | 8 | 0 | 105 | 38.1 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 5 | 12 | 4 | 76 | 8 | 0 | 105 | 20.0 | 0 | 0 |
| openai/gpt-4o-mini | 13 | 4 | 1 | 83 | 4 | 0 | 105 | 17.1 | 0 | 0 |
| google/gemini-2.5-flash | 32 | 37 | 3 | 28 | 5 | 0 | 105 | 68.6 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 12 | 34 | 1 | 55 | 3 | 0 | 105 | 44.8 | 0 | 0 |
| openai/gpt-5-mini | 42 | 7 | 1 | 40 | 15 | 0 | 105 | 47.6 | 0 | 0 |
| openai/gpt-5.5 | 78 | 20 | 0 | 7 | 0 | 0 | 105 | 93.3 | 0 | 0 |
