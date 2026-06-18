### Summary across all datasets

Methods ranked over **106 binary datasets**. *Avg. rank* is the mean across datasets of the per-dataset rank of each method's median MAE (rank 1 = best). *Median MAE* is the median of those per-dataset values across datasets. Lower is better in both.

| Method | Median MAE | Avg. rank |
|:---|---:|---:|
| QuantCal(knn,3) | 0.0000 | 2.71 |
| EMQ(bcts(rf)) | 0.0175 | 3.09 |
| DyS(rf) | 0.0177 | 3.53 |
| MS(rf) | 0.0209 | 4.95 |
| PACC(rf) | 0.0212 | 6.11 |
| EMQ(rf) | 0.0249 | 6.74 |
| FM(rf) | 0.0246 | 6.83 |
| ACC(rf) | 0.0251 | 7.54 |
| MLQ(rf) | 0.0253 | 8.66 |
| HDy(rf) | 0.0300 | 9.29 |
| KDEyML(rf) | 0.0532 | 9.46 |
| CC(rf) | 0.0600 | 11.03 |
| CC(bcts(rf)) | 0.0600 | 11.07 |

> Full per-dataset results: see [`RESULTS.md`](RESULTS.md).