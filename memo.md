# Breast Mass Classification: Logistic Regression and Decision Thresholds

## Part 1: From Scores to Decisions

This section uses the Explorer’s one-feature teaching example, which is separate from the two-feature dataset used later in the project. For each case, the score is `z = β₀ + β₁x`, the probability of class 1 is `p = 1 / (1 + exp(-z))`, and the predicted class is 1 when `p ≥ threshold`.

| Case | β₀ | β₁ | x | Threshold | Score z | Probability p | Predicted class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

**Case A calculation.** The score is `z = -1.0 + (0.8 × 2.0) = 0.60`. Applying the sigmoid gives `p = 1 / (1 + exp(-0.60)) ≈ 0.6457`. Since `0.6457 ≥ 0.50`, the predicted class is 1.

**A versus B.** Raising the threshold from 0.50 to 0.70 leaves the score and probability unchanged. Because `0.6457 < 0.70`, the predicted class changes from 1 to 0. The threshold changes the decision, not the probability curve.

**Reversing the coefficient.** In C, changing β₁ from 0.8 to -0.8 makes the probability curve decrease as x increases. At `x = 2.0`, the score falls to -2.60 and the probability of class 1 falls to 0.0691.

**Zero coefficient.** In D, β₁ is zero, so x contributes nothing to the score. For every value of x, `z = 1.0` and `p ≈ 0.7311`; the probability curve is flat.

The Explorer’s coefficients are set manually. Later in the project, the NumPy model will learn coefficients from training data by minimizing binary cross-entropy.
