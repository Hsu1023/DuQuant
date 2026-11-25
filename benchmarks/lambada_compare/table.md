### llama2

| Method | LAMBADA acc (%) | LAMBADA word PPL | WikiText-2 PPL | C4 PPL | LAMBADA six-task mean (%) |
|---|---:|---:|---:|---:|---:|
| fp16 | 73.92 | 3.395 | 5.472 | 7.263 | 67.05 |
| quarot | 70.83 | 4.084 | 6.430 | 8.875 | 63.99 |
| spinquant | 69.92 | 4.509 | 6.666 | 9.118 | 62.85 |
| duquant | 72.56 | 3.702 | 5.891 | 8.011 | 65.04 |
| flatquant | 72.91 | 3.576 | 5.774 | 7.799 | 65.72 |

### qwen3

| Method | LAMBADA acc (%) | LAMBADA word PPL | WikiText-2 PPL | C4 PPL | LAMBADA six-task mean (%) |
|---|---:|---:|---:|---:|---:|
| fp16 | 63.77 | 5.609 | 9.388 | 15.408 | 64.86 |
| quarot | 48.11 | 13.037 | 15.200 | 26.694 | 54.77 |
| spinquant | 48.55 | 13.349 | 14.180 | 24.198 | 55.92 |
| duquant | 51.52 | 10.820 | 12.361 | 20.995 | 58.56 |
| flatquant | 57.31 | 8.068 | 11.547 | 19.527 | 60.32 |
