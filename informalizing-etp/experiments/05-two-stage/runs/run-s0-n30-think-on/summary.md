# Benchmark run: seed=0, n=30, form=two-stage, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 8 | 22 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 849 |
| qwen/qwen3-32b | 18 | 10 | 0 | 2 | 0 | 0 | 30 | 93.3 | 30 | 686 |
| meta-llama/llama-3.3-70b-instruct | 15 | 14 | 0 | 1 | 0 | 0 | 30 | 96.7 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 17 | 10 | 0 | 3 | 0 | 0 | 30 | 90.0 | 0 | 0 |
| openai/gpt-4o-mini | 26 | 0 | 0 | 2 | 2 | 0 | 30 | 86.7 | 0 | 0 |
| google/gemini-2.5-flash | 23 | 6 | 0 | 1 | 0 | 0 | 30 | 96.7 | 30 | 1764 |
| anthropic/claude-haiku-4.5 | 29 | 1 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 514 |
| openai/gpt-5-mini | 27 | 3 | 0 | 0 | 0 | 0 | 30 | 100.0 | 30 | 768 |
