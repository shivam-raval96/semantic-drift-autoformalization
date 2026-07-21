# Benchmark run: seed=0, n=30, form=literal, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 34 | 6 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 983 |
| qwen/qwen3-32b | 31 | 8 | 0 | 0 | 1 | 0 | 40 | 97.5 | 40 | 767 |
| meta-llama/llama-3.3-70b-instruct | 40 | 0 | 0 | 0 | 0 | 0 | 40 | 100.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 29 | 9 | 0 | 1 | 1 | 0 | 40 | 95.0 | 0 | 0 |
| openai/gpt-4o-mini | 40 | 0 | 0 | 0 | 0 | 0 | 40 | 100.0 | 0 | 0 |
| google/gemini-2.5-flash | 34 | 6 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 1183 |
| anthropic/claude-haiku-4.5 | 38 | 2 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 544 |
| openai/gpt-5-mini | 38 | 2 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 768 |
