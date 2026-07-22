# Benchmark run: seed=0, n=30, form=story, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 96 | 31 | 0 | 29 | 4 | 0 | 160 | 79.4 | 0 | 0 |
| qwen/qwen3-32b | 30 | 25 | 5 | 97 | 3 | 0 | 160 | 37.5 | 140 | 1 |
| meta-llama/llama-3.3-70b-instruct | 33 | 43 | 1 | 70 | 13 | 0 | 160 | 48.1 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 38 | 22 | 2 | 81 | 17 | 0 | 160 | 38.8 | 0 | 0 |
| openai/gpt-4o-mini | 40 | 8 | 3 | 106 | 3 | 0 | 160 | 31.9 | 0 | 0 |
| google/gemini-2.5-flash | 90 | 33 | 5 | 31 | 1 | 0 | 160 | 80.0 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 39 | 46 | 2 | 67 | 6 | 0 | 160 | 54.4 | 0 | 0 |
| openai/gpt-5-mini | 71 | 14 | 1 | 57 | 17 | 0 | 160 | 53.8 | 0 | 0 |
| openai/gpt-5.5 | 132 | 20 | 0 | 8 | 0 | 0 | 160 | 95.0 | 0 | 0 |
