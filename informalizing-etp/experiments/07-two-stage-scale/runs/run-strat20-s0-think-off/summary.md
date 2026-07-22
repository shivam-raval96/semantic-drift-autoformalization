# Benchmark run: seed=0, n=30, form=two-stage, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 54 | 63 | 1 | 37 | 5 | 0 | 160 | 73.8 | 0 | 0 |
| qwen/qwen3-32b | 18 | 43 | 1 | 94 | 4 | 0 | 160 | 38.8 | 144 | 1 |
| meta-llama/llama-3.3-70b-instruct | 60 | 34 | 0 | 66 | 0 | 0 | 160 | 58.8 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 12 | 68 | 0 | 74 | 6 | 0 | 160 | 50.0 | 0 | 0 |
| openai/gpt-4o-mini | 19 | 17 | 1 | 113 | 10 | 0 | 160 | 23.1 | 0 | 0 |
| google/gemini-2.5-flash | 93 | 25 | 0 | 36 | 6 | 0 | 160 | 73.8 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 40 | 64 | 0 | 54 | 2 | 0 | 160 | 65.0 | 0 | 0 |
| openai/gpt-5-mini | 73 | 18 | 0 | 42 | 27 | 0 | 160 | 56.9 | 0 | 0 |
| openai/gpt-5.5 | 159 | 0 | 0 | 1 | 0 | 0 | 160 | 99.4 | 0 | 0 |
