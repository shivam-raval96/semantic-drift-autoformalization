# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 34 | 3 | 0 | 3 | 0 | 0 | 40 | 92.5 | 0 | 0 |
| qwen/qwen3-32b | 37 | 1 | 0 | 2 | 0 | 0 | 40 | 95.0 | 23 | 1 |
| meta-llama/llama-3.3-70b-instruct | 35 | 0 | 0 | 5 | 0 | 0 | 40 | 87.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 32 | 0 | 0 | 8 | 0 | 0 | 40 | 80.0 | 0 | 0 |
| openai/gpt-4o-mini | 39 | 0 | 0 | 1 | 0 | 0 | 40 | 97.5 | 0 | 0 |
| google/gemini-2.5-flash | 39 | 0 | 0 | 1 | 0 | 0 | 40 | 97.5 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 37 | 0 | 0 | 2 | 1 | 0 | 40 | 92.5 | 0 | 0 |
| openai/gpt-5-mini | 38 | 0 | 0 | 2 | 0 | 0 | 40 | 95.0 | 0 | 0 |
