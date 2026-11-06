# Frugal NLP leaderboard

_generated 2026-10-29T18:54:16+00:00 on NVIDIA GeForce RTX 3050 A Laptop GPU_

| use_case   | model_id                                          | kind       | base_model                 |   n_train |   macro_f1 |   accuracy |   wh_per_1k |   latency_ms |   train_wh |   eval_g_co2 |   data_left_premises |   trainable_params |   n_test |   rank_f1 |
|:-----------|:--------------------------------------------------|:-----------|:---------------------------|----------:|-----------:|-----------:|------------:|-------------:|-----------:|-------------:|---------------------:|-------------------:|---------:|----------:|
| sentiment  | llm-gemini-gemini-3.5-flash-lite-sentiment        | llm-gemini | gemini-3.5-flash-lite      |         0 |     0.9716 |     0.97   |     10.0261 |        33.3  |     0      |      0.45121 |                    1 |                  0 |      300 |         1 |
| sentiment  | llm-local-Qwen2.5-0.5B-Instruct-sentiment         | llm-local  | Qwen/Qwen2.5-0.5B-Instruct |         0 |     0.9033 |     0.9033 |      0.2269 |         8.84 |     0      |      0.04857 |                    0 |                  0 |      300 |         2 |
| sentiment  | slm-sentiment-bge-base-en-v1.5-32pc-73a641        | slm        | BAAI/bge-base-en-v1.5      |        64 |     0.8499 |     0.85   |      0.0617 |         1.99 |     0.1088 |      0.0132  |                    0 |             296450 |      300 |         3 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-64pc-cbe69d       | slm        | BAAI/bge-small-en-v1.5     |       128 |     0.8333 |     0.8333 |      0.0603 |         1.21 |     0.14   |      0.01291 |                    0 |             148226 |      300 |         4 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-32pc-a0e6bb       | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.8257 |     0.8267 |      0.0171 |         1.26 |     0.0711 |      0.00367 |                    0 |             148226 |      300 |         5 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-32pc-1e03a9       | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.8233 |     0.8233 |      0.0515 |         1.07 |     0.1062 |      0.01102 |                    0 |             148226 |      300 |         6 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-32pc-025c43       | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.82   |     0.82   |      0.0658 |         1.25 |     0.0771 |      0.01408 |                    0 |             148226 |      300 |         7 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-16pc-f7402f       | slm        | BAAI/bge-small-en-v1.5     |        32 |     0.8097 |     0.81   |      0.0231 |         1.1  |     0.0447 |      0.00494 |                    0 |             148226 |      300 |         8 |
| sentiment  | slm-sentiment-bge-small-en-v1.5-8pc-9ed6fd        | slm        | BAAI/bge-small-en-v1.5     |        16 |     0.8071 |     0.81   |      0.0437 |         1.03 |     0.0308 |      0.00935 |                    0 |             148226 |      300 |         9 |
| sentiment  | slm-sentiment-distilroberta-base-32pc-bf35fb      | slm        | distilroberta-base         |        64 |     0.7732 |     0.7733 |      0.0456 |         1.31 |     0.0664 |      0.00976 |                    0 |             739586 |      300 |        10 |
| sentiment  | slm-sentiment-distilbert-base-uncased-32pc-3862d2 | slm        | distilbert-base-uncased    |        64 |     0.7533 |     0.7533 |      0.0477 |         1.28 |     0.066  |      0.01022 |                    0 |             739586 |      300 |        11 |
| spam       | llm-gemini-gemini-3.5-flash-lite-spam             | llm-gemini | gemini-3.5-flash-lite      |         0 |     0.9729 |     0.9867 |     11.7706 |       162.54 |     0      |      0.82458 |                    1 |                  0 |      300 |         1 |
| spam       | slm-spam-distilbert-base-uncased-32pc-1e84cf      | slm        | distilbert-base-uncased    |        64 |     0.9593 |     0.98   |      0.0576 |         1.53 |     0.0754 |      0.01232 |                    0 |             739586 |      300 |         2 |
| spam       | slm-spam-bge-small-en-v1.5-16pc-8f6890            | slm        | BAAI/bge-small-en-v1.5     |        32 |     0.9529 |     0.9767 |      0.0298 |         1.35 |     0.0455 |      0.00638 |                    0 |             148226 |      300 |         3 |
| spam       | slm-spam-bge-small-en-v1.5-32pc-c889e8            | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.9406 |     0.97   |      0.0326 |         1.02 |     0.1157 |      0.00697 |                    0 |             148226 |      300 |         4 |
| spam       | slm-spam-bge-small-en-v1.5-32pc-866302            | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.9346 |     0.9667 |      0.0529 |         1.39 |     0.0795 |      0.01133 |                    0 |             148226 |      300 |         5 |
| spam       | slm-spam-bge-small-en-v1.5-64pc-434fb7            | slm        | BAAI/bge-small-en-v1.5     |       128 |     0.9243 |     0.96   |      0.0333 |         1.02 |     0.1825 |      0.00714 |                    0 |             148226 |      300 |         6 |
| spam       | slm-spam-bge-small-en-v1.5-16pc-565dec            | slm        | BAAI/bge-small-en-v1.5     |        32 |     0.9193 |     0.96   |      0.0948 |         1.4  |     0.0285 |      0.01353 |                    0 |             148226 |      200 |         7 |
| spam       | slm-spam-bge-small-en-v1.5-32pc-a8639c            | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.9173 |     0.9567 |      0.0491 |         1.38 |     0.0776 |      0.01052 |                    0 |             148226 |      300 |         8 |
| spam       | slm-spam-bge-small-en-v1.5-8pc-aeb170             | slm        | BAAI/bge-small-en-v1.5     |        16 |     0.7502 |     0.8267 |      0.0236 |         1.04 |     0.0276 |      0.00506 |                    0 |             148226 |      300 |         9 |
| spam       | llm-local-Qwen2.5-0.5B-Instruct-spam              | llm-local  | Qwen/Qwen2.5-0.5B-Instruct |         0 |     0.488  |     0.8667 |      0.2195 |        10.36 |     0      |      0.04698 |                    0 |                  0 |      300 |        10 |
| topic      | llm-gemini-gemini-3.5-flash-lite-topic            | llm-gemini | gemini-3.5-flash-lite      |         0 |     0.894  |     0.8867 |     11.5244 |       152.52 |     0      |      0.77188 |                    1 |                  0 |      300 |         1 |
| topic      | slm-topic-bge-base-en-v1.5-32pc-6aae4c            | slm        | BAAI/bge-base-en-v1.5      |       128 |     0.8828 |     0.8767 |      0.1073 |         3.05 |     0.3467 |      0.02296 |                    0 |             297988 |      300 |         2 |
| topic      | slm-topic-bge-small-en-v1.5-64pc-1ebe2b           | slm        | BAAI/bge-small-en-v1.5     |       256 |     0.8728 |     0.8633 |      0.0302 |         1.72 |     0.3425 |      0.00647 |                    0 |             148996 |      300 |         3 |
| topic      | slm-topic-distilroberta-base-32pc-19f2ee          | slm        | distilroberta-base         |       128 |     0.8601 |     0.8533 |      0.0319 |         1.86 |     0.1705 |      0.00683 |                    0 |             741124 |      300 |         4 |
| topic      | slm-topic-bge-small-en-v1.5-16pc-e04599           | slm        | BAAI/bge-small-en-v1.5     |        64 |     0.8581 |     0.85   |      0.0733 |         1.49 |     0.0837 |      0.01568 |                    0 |             148996 |      300 |         5 |
| topic      | slm-topic-bge-small-en-v1.5-32pc-92e54d           | slm        | BAAI/bge-small-en-v1.5     |       128 |     0.8572 |     0.8467 |      0.0642 |         1.59 |     0.1658 |      0.01374 |                    0 |             148996 |      300 |         6 |
| topic      | slm-topic-distilbert-base-uncased-32pc-978952     | slm        | distilbert-base-uncased    |       128 |     0.8547 |     0.8467 |      0.0534 |         2.27 |     0.1691 |      0.01143 |                    0 |             741124 |      300 |         7 |
| topic      | slm-topic-bge-small-en-v1.5-8pc-3b0aa0            | slm        | BAAI/bge-small-en-v1.5     |        32 |     0.78   |     0.7733 |      0.0314 |         1.7  |     0.0389 |      0.00672 |                    0 |             148996 |      300 |         8 |
| topic      | llm-local-Qwen2.5-0.5B-Instruct-topic             | llm-local  | Qwen/Qwen2.5-0.5B-Instruct |         0 |     0.4284 |     0.5033 |      0.3736 |        14.42 |     0      |      0.07996 |                    0 |                  0 |      300 |         9 |

## Energy break-even: sentiment

| slm                                               | llm                                        |   slm_train_wh |   breakeven_predictions |   energy_ratio_llm_over_slm |
|:--------------------------------------------------|:-------------------------------------------|---------------:|------------------------:|----------------------------:|
| slm-sentiment-bge-base-en-v1.5-32pc-73a641        | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.1088 |                      11 |                       162.5 |
| slm-sentiment-bge-base-en-v1.5-32pc-73a641        | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.1088 |                     659 |                         3.7 |
| slm-sentiment-bge-small-en-v1.5-64pc-cbe69d       | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.14   |                      15 |                       166.3 |
| slm-sentiment-bge-small-en-v1.5-64pc-cbe69d       | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.14   |                     841 |                         3.8 |
| slm-sentiment-bge-small-en-v1.5-32pc-a0e6bb       | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.0711 |                       8 |                       586.3 |
| slm-sentiment-bge-small-en-v1.5-32pc-a0e6bb       | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.0711 |                     339 |                        13.3 |
| slm-sentiment-bge-small-en-v1.5-32pc-1e03a9       | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.1062 |                      11 |                       194.7 |
| slm-sentiment-bge-small-en-v1.5-32pc-1e03a9       | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.1062 |                     606 |                         4.4 |
| slm-sentiment-bge-small-en-v1.5-32pc-025c43       | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.0771 |                       8 |                       152.4 |
| slm-sentiment-bge-small-en-v1.5-32pc-025c43       | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.0771 |                     479 |                         3.4 |
| slm-sentiment-bge-small-en-v1.5-16pc-f7402f       | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.0447 |                       5 |                       434   |
| slm-sentiment-bge-small-en-v1.5-16pc-f7402f       | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.0447 |                     220 |                         9.8 |
| slm-sentiment-bge-small-en-v1.5-8pc-9ed6fd        | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.0308 |                       4 |                       229.4 |
| slm-sentiment-bge-small-en-v1.5-8pc-9ed6fd        | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.0308 |                     169 |                         5.2 |
| slm-sentiment-distilroberta-base-32pc-bf35fb      | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.0664 |                       7 |                       219.9 |
| slm-sentiment-distilroberta-base-32pc-bf35fb      | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.0664 |                     367 |                         5   |
| slm-sentiment-distilbert-base-uncased-32pc-3862d2 | llm-gemini-gemini-3.5-flash-lite-sentiment |         0.066  |                       7 |                       210.2 |
| slm-sentiment-distilbert-base-uncased-32pc-3862d2 | llm-local-Qwen2.5-0.5B-Instruct-sentiment  |         0.066  |                     369 |                         4.8 |

## Energy break-even: spam

| slm                                          | llm                                   |   slm_train_wh |   breakeven_predictions |   energy_ratio_llm_over_slm |
|:---------------------------------------------|:--------------------------------------|---------------:|------------------------:|----------------------------:|
| slm-spam-distilbert-base-uncased-32pc-1e84cf | llm-gemini-gemini-3.5-flash-lite-spam |         0.0754 |                       7 |                       204.4 |
| slm-spam-distilbert-base-uncased-32pc-1e84cf | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0754 |                     466 |                         3.8 |
| slm-spam-bge-small-en-v1.5-16pc-8f6890       | llm-gemini-gemini-3.5-flash-lite-spam |         0.0455 |                       4 |                       395   |
| slm-spam-bge-small-en-v1.5-16pc-8f6890       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0455 |                     240 |                         7.4 |
| slm-spam-bge-small-en-v1.5-32pc-c889e8       | llm-gemini-gemini-3.5-flash-lite-spam |         0.1157 |                      10 |                       361.1 |
| slm-spam-bge-small-en-v1.5-32pc-c889e8       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.1157 |                     620 |                         6.7 |
| slm-spam-bge-small-en-v1.5-32pc-866302       | llm-gemini-gemini-3.5-flash-lite-spam |         0.0795 |                       7 |                       222.5 |
| slm-spam-bge-small-en-v1.5-32pc-866302       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0795 |                     478 |                         4.1 |
| slm-spam-bge-small-en-v1.5-64pc-434fb7       | llm-gemini-gemini-3.5-flash-lite-spam |         0.1825 |                      16 |                       353.5 |
| slm-spam-bge-small-en-v1.5-64pc-434fb7       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.1825 |                     981 |                         6.6 |
| slm-spam-bge-small-en-v1.5-16pc-565dec       | llm-gemini-gemini-3.5-flash-lite-spam |         0.0285 |                       3 |                       124.2 |
| slm-spam-bge-small-en-v1.5-16pc-565dec       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0285 |                     229 |                         2.3 |
| slm-spam-bge-small-en-v1.5-32pc-a8639c       | llm-gemini-gemini-3.5-flash-lite-spam |         0.0776 |                       7 |                       239.7 |
| slm-spam-bge-small-en-v1.5-32pc-a8639c       | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0776 |                     456 |                         4.5 |
| slm-spam-bge-small-en-v1.5-8pc-aeb170        | llm-gemini-gemini-3.5-flash-lite-spam |         0.0276 |                       3 |                       498.8 |
| slm-spam-bge-small-en-v1.5-8pc-aeb170        | llm-local-Qwen2.5-0.5B-Instruct-spam  |         0.0276 |                     141 |                         9.3 |

## Energy break-even: topic

| slm                                           | llm                                    |   slm_train_wh |   breakeven_predictions |   energy_ratio_llm_over_slm |
|:----------------------------------------------|:---------------------------------------|---------------:|------------------------:|----------------------------:|
| slm-topic-bge-base-en-v1.5-32pc-6aae4c        | llm-gemini-gemini-3.5-flash-lite-topic |         0.3467 |                      31 |                       107.4 |
| slm-topic-bge-base-en-v1.5-32pc-6aae4c        | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.3467 |                    1302 |                         3.5 |
| slm-topic-bge-small-en-v1.5-64pc-1ebe2b       | llm-gemini-gemini-3.5-flash-lite-topic |         0.3425 |                      30 |                       381.6 |
| slm-topic-bge-small-en-v1.5-64pc-1ebe2b       | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.3425 |                     998 |                        12.4 |
| slm-topic-distilroberta-base-32pc-19f2ee      | llm-gemini-gemini-3.5-flash-lite-topic |         0.1705 |                      15 |                       361.3 |
| slm-topic-distilroberta-base-32pc-19f2ee      | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.1705 |                     499 |                        11.7 |
| slm-topic-bge-small-en-v1.5-16pc-e04599       | llm-gemini-gemini-3.5-flash-lite-topic |         0.0837 |                       8 |                       157.2 |
| slm-topic-bge-small-en-v1.5-16pc-e04599       | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.0837 |                     279 |                         5.1 |
| slm-topic-bge-small-en-v1.5-32pc-92e54d       | llm-gemini-gemini-3.5-flash-lite-topic |         0.1658 |                      15 |                       179.5 |
| slm-topic-bge-small-en-v1.5-32pc-92e54d       | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.1658 |                     536 |                         5.8 |
| slm-topic-distilbert-base-uncased-32pc-978952 | llm-gemini-gemini-3.5-flash-lite-topic |         0.1691 |                      15 |                       215.8 |
| slm-topic-distilbert-base-uncased-32pc-978952 | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.1691 |                     529 |                         7   |
| slm-topic-bge-small-en-v1.5-8pc-3b0aa0        | llm-gemini-gemini-3.5-flash-lite-topic |         0.0389 |                       4 |                       367   |
| slm-topic-bge-small-en-v1.5-8pc-3b0aa0        | llm-local-Qwen2.5-0.5B-Instruct-topic  |         0.0389 |                     114 |                        11.9 |