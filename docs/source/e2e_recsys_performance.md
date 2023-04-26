## End-to-End RecSys Democratization Performance

Performance results are evaluated on 4-node cluster configured with Intel(R) Xeon(R) Platinum 8358 Scalable processor.
For [WnD](../../modelzoo/WnD/README.md), [DIEN](../../modelzoo/dien/README.md) and [DLRM](../../modelzoo/dlrm/README.md), Intel® End-to-End AI Optimization Kit delivered 51.01x(5.02x ELT & 113.03x training), 12.67x(14.86x ELT & 11.91x training) and 71.16x(86.40x ELT & 42.31x training) E2E time speedup, 21.18x, 14.11x and 124.98x inference throughput speedup respectively. Please refer to corresponding model link for detailed test dataset and test method.

| Model | ETL | Training | Inference |
| ----- | --- | -------- | --------- |
| DLRM | 86.40 | 42.31 | 124.98 |
| DIEN | 14.86 | 11.91 | 14.11 |
| WnD | 5.02 | 113.03 | 21.18 |

<img src="./e2eaiok_v01_performance.png" width="500"/>