# Benchmark report

Total records: 24
  ok:      14
  skipped: 9
  error:   0
  timeout: 1

## Median wall time (ms) per (impl × fixture × batch) at CPU

```
cfg_impl                    multitorch_cached  multitorch_from_scratch    pyctm  pyttmult
cfg_fixture cfg_batch_size                                                               
ni2_d8_oh   1                           10.12                  1215.71    35.22     34.42
            10                          92.30                 12053.69   341.89    338.35
            100                        945.55                      NaN  3489.37   3493.22
nid8ct      1                           51.63                      NaN      NaN       NaN
            10                         508.59                      NaN      NaN       NaN
            100                       5588.93                      NaN      NaN       NaN
```

## Plots

- `p1_single_spectrum_time.png` — Single-spectrum time per impl × fixture (batch=1, CPU)
- `p2_scaling_vs_batch.png` — Scaling vs batch size, faceted by fixture
- `p3_parity_cosine.png` — Parity cosine similarity heatmap
- `p4_cpu_vs_cuda.png` — multitorch CPU-vs-CUDA speedup (if CUDA data present)