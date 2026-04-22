# Benchmark report

Total records: 1158
  ok:      186
  skipped: 705
  error:   37
  timeout: 230

## Median wall time (ms) per (impl × fixture × batch) at CPU

```
cfg_impl                    multitorch_batch  multitorch_cached  multitorch_from_scratch     pyctm  pyttmult  ttmult_raw
cfg_fixture cfg_batch_size                                                                                              
als1ni2     1                            NaN             145.41                      NaN       NaN       NaN         NaN
            10                       1605.61            1449.60                      NaN       NaN       NaN         NaN
            100                     14599.29           14699.85                      NaN       NaN       NaN         NaN
co2_d7_oh   1                            NaN             166.38                  4104.24     45.85     45.45       43.99
            10                       1872.14            1686.65                      NaN    441.91    479.72      416.16
            100                     17001.95           16779.50                      NaN   4659.23   4636.15     4263.29
            1000                         NaN                NaN                      NaN  47030.85  46877.64    43991.67
cr3_d3_oh   1                            NaN                NaN                  7770.43    141.26    141.24      147.31
            10                           NaN                NaN                      NaN   1062.37   1050.59     1043.89
            100                          NaN                NaN                      NaN  11030.51  10931.28    10699.43
fe2_d6_oh   1                            NaN             299.76                  8956.89     82.06     74.54       70.20
            10                       3315.59            3128.09                      NaN    801.32    795.13      762.27
            100                          NaN                NaN                      NaN   7961.85   7933.34     7478.94
fe3_d5_oh   1                            NaN            7033.47                 10144.58    129.80    119.77      112.00
            10                           NaN                NaN                      NaN   1259.00   1250.77     1150.92
            100                          NaN                NaN                      NaN  12402.48  12398.62    11373.62
mn2_d5_oh   1                            NaN            7019.35                  9981.17    125.54    117.05      112.47
            10                           NaN                NaN                      NaN   1272.11   1225.84     1148.14
            100                          NaN                NaN                      NaN  12197.87  12297.02    11361.22
ni2_d8_oh   1                            NaN              28.32                  2071.90     39.15     43.78       37.62
            10                        315.69             286.66                 20624.08    384.24    381.00      361.60
            100                      2752.18            2791.49                      NaN   3811.37   3821.42     3534.31
            1000                    25989.73           27587.28                      NaN  37668.56  34349.77    33340.37
nid8        1                            NaN             143.91                      NaN       NaN     40.56       39.10
            10                       1608.53            1485.94                      NaN       NaN    404.51      363.80
            100                     14572.83           14662.90                      NaN       NaN   3988.70     3720.86
            1000                         NaN                NaN                      NaN       NaN  40129.72    37638.75
nid8ct      1                            NaN             143.20                      NaN       NaN       NaN         NaN
            10                       1606.74            1491.76                      NaN       NaN       NaN         NaN
            100                     14731.29           14533.75                      NaN       NaN       NaN         NaN
ti4_d0_oh   1                            NaN              19.80                      NaN     33.58     39.88       37.37
            10                        216.21             195.43                      NaN    378.66    354.69      371.83
            100                      1934.63            2030.08                      NaN   4101.41   4052.40     3815.83
            1000                    15257.88           18648.57                      NaN  29722.76  29812.94    30704.77
v3_d2_oh    1                            NaN            2356.73                  6368.49     72.16     75.17       72.26
            10                      24241.23           23137.06                      NaN    697.91    693.30      887.03
            100                          NaN                NaN                      NaN   6652.96   6642.74     8021.63
```

## Plots

- `p1_single_spectrum_time.png` — Single-spectrum time per impl × fixture (batch=1, CPU)
- `p2_scaling_vs_batch.png` — Scaling vs batch size, faceted by fixture
- `p3_parity_cosine.png` — Parity cosine similarity heatmap
- `p4_cpu_vs_cuda.png` — multitorch CPU-vs-CUDA speedup (if CUDA data present)