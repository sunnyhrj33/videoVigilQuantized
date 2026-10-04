# Open-set evaluation (CHIRLA)

- Re-ID model: `osnet_ain_x1_0` (OSNet-AIN, MSMT17 weights)
- Protocol: same-day ('morning entrance') enrollment; each tracklet matched against its own day's gallery
- Enrolled per day: 2023-06-02: 6, 2023-06-08: 7, 2023-06-13: 6, 2023-07-26: 10, 2023-12-01: 8, 2023-12-04: 7, 2023-12-05: 12
- Train days: 2023-06-02, 2023-06-08, 2023-06-13  (visitors [8, 18, 19, 24])
- Test days: 2023-07-26, 2023-12-01, 2023-12-04, 2023-12-05  (visitors [13, 15, 16, 25], unseen in training)
- Tracklets: 10 crops max, 4 min; employee footage before enrollment cut-off + 20 s excluded
- Tracklets used: train 259, test 601
- P(Known) threshold: 0.5; naive cosine threshold: 0.45

TAR = employee accepted with correct ID · MisID = employee accepted as wrong ID · FAR = visitor accepted as an employee · TRR = visitor rejected as UNKNOWN

| Method (test split) | TAR ↑ | MisID ↓ | FAR ↓ | TRR ↑ | samples emp / vis |
|---|---|---|---|---|---|
| Naive single-frame, closed-set (always nearest) | 62.4% | 37.6% | 100.0% | 0.0% | 2901 / 298 |
| Naive single-frame, cosine ≥ 0.45 | 62.3% | 36.7% | 98.7% | 1.3% | 2901 / 298 |
| Tracklet + rule-based fallback | 73.0% | 18.4% | 92.3% | 7.7% | 549 / 52 |
| **Tracklet + logistic regression** | 66.7% | 6.2% | 26.9% | 73.1% | 549 / 52 |

Tracklet LR ROC-AUC (test): 0.881

Logistic-regression weights (standardised features):

- s_best: +0.581
- margin: +1.343
- agreement: -0.107
- consistency: -0.127
- prototype_fit: +0.665
- intercept: +0.873