# Open-set evaluation (CHIRLA)

- Re-ID model: `osnet_ain_x1_0` (OSNet-AIN, MSMT17 weights)
- Protocol: same-day ('morning entrance') enrollment; each tracklet matched against its own day's gallery
- Enrolled per day: 2023-06-02: 6, 2023-06-08: 7, 2023-06-13: 6, 2023-07-26: 11, 2023-12-01: 8, 2023-12-04: 7, 2023-12-05: 12
- Train days: 2023-06-02, 2023-06-08, 2023-06-13  (visitors [8, 18, 19, 24])
- Test days: 2023-07-26, 2023-12-01, 2023-12-04, 2023-12-05  (visitors [13, 15, 16, 25], unseen in training)
- Tracklets: 10 crops max, 4 min; employee footage before enrollment cut-off + 20 s excluded
- Tracklets used: train 232, test 554
- P(Known) threshold: 0.5; naive cosine threshold: 0.45

TAR = employee accepted with correct ID · MisID = employee accepted as wrong ID · FAR = visitor accepted as an employee · TRR = visitor rejected as UNKNOWN

| Method (test split) | TAR ↑ | MisID ↓ | FAR ↓ | TRR ↑ | samples emp / vis |
|---|---|---|---|---|---|
| Naive single-frame, closed-set (always nearest) | 62.0% | 38.0% | 100.0% | 0.0% | 2656 / 298 |
| Naive single-frame, cosine ≥ 0.45 | 62.0% | 37.5% | 99.7% | 0.3% | 2656 / 298 |
| Tracklet + rule-based fallback | 74.5% | 17.5% | 88.5% | 11.5% | 502 / 52 |
| **Tracklet + logistic regression** | 63.7% | 3.4% | 30.8% | 69.2% | 502 / 52 |

Tracklet LR ROC-AUC (test): 0.900

Logistic-regression weights (standardised features):

- s_best: +0.723
- margin: +0.428
- agreement: -0.087
- consistency: +0.172
- prototype_fit: +0.970
- intercept: +0.380