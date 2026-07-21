# Benchmark run: seed=0, n=30, form=literal, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 23 | 6 | 0 | 0 | 0 | 1 | 29 | 100.0 | 30 | 799 |
| qwen/qwen3-32b | 26 | 4 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 664 |
| meta-llama/llama-3.3-70b-instruct | 28 | 0 | 0 | 0 | 2 | 0 | 30 | 93.3 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 26 | 4 | 0 | 0 | 0 | 0 | 30 | 100.0 | 0 | 0 |
| openai/gpt-4o-mini | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 0 | 0 |
| google/gemini-2.5-flash | 26 | 4 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 1474 |
| anthropic/claude-haiku-4.5 | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 493 |
| openai/gpt-5-mini | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 672 |
