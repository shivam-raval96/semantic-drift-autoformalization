# Benchmark run: seed=0, n=30, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 986 |
| qwen/qwen3-32b | 24 | 6 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 1039 |
| meta-llama/llama-3.3-70b-instruct | 17 | 12 | 0 | 1 | 0 | 0 | 30 | 96.7 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 26 | 2 | 0 | 2 | 0 | 0 | 30 | 93.3 | 0 | 0 |
| openai/gpt-4o-mini | 21 | 0 | 0 | 7 | 2 | 0 | 30 | 70.0 | 0 | 0 |
| google/gemini-2.5-flash | 28 | 2 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 1865 |
| anthropic/claude-haiku-4.5 | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 592 |
| openai/gpt-5-mini | 28 | 2 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 1152 |
