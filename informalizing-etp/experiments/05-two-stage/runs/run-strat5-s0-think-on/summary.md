# Benchmark run: seed=0, n=30, form=two-stage, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 13 | 26 | 1 | 0 | 0 | 0 | 40 | 100.0 | 40 | 929 |
| qwen/qwen3-32b | 20 | 9 | 2 | 8 | 1 | 0 | 40 | 77.5 | 40 | 765 |
| meta-llama/llama-3.3-70b-instruct | 19 | 16 | 1 | 4 | 0 | 0 | 40 | 90.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 17 | 14 | 0 | 9 | 0 | 0 | 40 | 77.5 | 0 | 0 |
| openai/gpt-4o-mini | 21 | 1 | 0 | 17 | 1 | 0 | 40 | 55.0 | 0 | 0 |
| google/gemini-2.5-flash | 30 | 10 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 1853 |
| anthropic/claude-haiku-4.5 | 35 | 2 | 0 | 3 | 0 | 0 | 40 | 92.5 | 40 | 471 |
| openai/gpt-5-mini | 35 | 5 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 768 |
