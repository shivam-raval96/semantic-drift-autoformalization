# Benchmark run: seed=0, n=30, form=literal, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 29 | 11 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 1111 |
| qwen/qwen3-32b | 33 | 4 | 0 | 2 | 1 | 0 | 40 | 92.5 | 40 | 950 |
| meta-llama/llama-3.3-70b-instruct | 33 | 2 | 0 | 5 | 0 | 0 | 40 | 87.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 30 | 4 | 0 | 6 | 0 | 0 | 40 | 85.0 | 0 | 0 |
| openai/gpt-4o-mini | 25 | 5 | 0 | 6 | 4 | 0 | 40 | 75.0 | 0 | 0 |
| google/gemini-2.5-flash | 38 | 0 | 0 | 2 | 0 | 0 | 40 | 95.0 | 40 | 1516 |
| anthropic/claude-haiku-4.5 | 38 | 0 | 0 | 2 | 0 | 0 | 40 | 95.0 | 40 | 708 |
| openai/gpt-5-mini | 36 | 4 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 992 |
