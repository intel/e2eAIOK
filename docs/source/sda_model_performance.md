## SDA Model Performance

Performance results are evaluated on 4-node cluster configured with Intel(R) Xeon(R) Platinum 8358 Scalable processor.
For [MiniGO](../../modelzoo/minigo/README.md), [BERT](../../modelzoo/bert/README.md), [ResNet](../../modelzoo/resnet/README.md), [RNN-T](../../modelzoo/rnnt/README.md), Intel® End-to-End AI Optimization Kit delivered 13.06x, 10.10x, 10.12x and 11.61x training time speedup respecitvely through E2E optimizations. Please refer to corresponding model link for detailed test dataset and test method. 
> Noted: Optimized lighter models' accuracy are slightly lower: ResNet -5% accuracy, BERT -1% F1 score, RNN-T -1% WER.

| Model | Training | Accuracy Ratio |
| ----- | ---------| -------------- |
| ResNet | 10.12 | -5% |
| BERT | 10.10 | -1% |
| RNN-T | 11.61 | -1% |
| MiniGo | 13.06 | 0% |

<img src="./e2eaiok_v02_performance.png" width="500"/>