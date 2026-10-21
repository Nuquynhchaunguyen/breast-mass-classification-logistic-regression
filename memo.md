# Breast Mass Classification: Logistic Regression and Decision Thresholds

## Part 1: From Scores to Decisions

The one-feature Explorer is a teaching example separate from the dataset used below. Its score is `z = β₀ + β₁x`; the sigmoid converts that score to `p = 1 / (1 + exp(-z))`. A case is classified as 1 when `p ≥ threshold`.

| Case | β₀ | β₁ | x | Threshold | Score z | Probability p | Predicted class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

For A, `z = -1.0 + 0.8 × 2.0 = 0.60`, so `p = 1 / (1 + exp(-0.60)) ≈ 0.6457`; this exceeds 0.50. B has the same score and probability, but raising the threshold to 0.70 changes its predicted class to 0. In C, reversing the coefficient makes probability decrease as `x` increases. In D, the zero coefficient removes the effect of `x`, leaving a constant probability. I set these Explorer coefficients manually; the NumPy model below learns its coefficients from training data.

## Part 2: Data Preparation

I used the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn, originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). I selected only `mean radius` and `mean texture`, in that order. The 569 observations have no missing values in those features. I recoded the target to **1 = malignant** (positive class) and **0 = benign**. The dataset contains 212 malignant and 357 benign cases.

| Split | Total | Benign (0) | Malignant (1) |
|---|---:|---:|---:|
| Training | 341 | 214 | 127 |
| Validation | 114 | 71 | 43 |
| Test | 114 | 72 | 42 |

Stratified splits with `random_state=42` preserved similar class proportions. Checks found no overlapping rows, and all 569 observations were assigned. I fitted `StandardScaler` on training rows only and applied that fitted scaler to validation and test rows. This prevents those held-out rows from influencing preprocessing.

## Part 3: NumPy Logistic Regression

My NumPy model uses an intercept and the two standardized features. All coefficients began at zero. Full-batch gradient descent used learning rate `0.1` for `20,000` updates, without regularization or class weights. Minimizing mean binary cross-entropy increases the likelihood assigned to the observed labels. Numerically stable sigmoid and loss calculations avoided overflow.

Training loss fell from `0.693147` to `0.246834`, then leveled off in `reports/training_loss.png`. A coefficient describes a change in **log-odds**, not a fixed change in probability: holding texture fixed, a one-training-standard-deviation increase in mean radius is associated with about `3.6568` higher log-odds of malignancy. This is an association in the fitted model, not a causal effect.

## Part 4: Reference Model and Baseline

I fitted unregularized scikit-learn logistic regression on the same scaled training rows. The coefficients closely match:

| Coefficient | NumPy | scikit-learn |
|---|---:|---:|
| Intercept | -0.7271724 | -0.7271724 |
| Mean radius, standardized | 3.6567942 | 3.6567942 |
| Mean texture, standardized | 1.0098715 | 1.0098715 |

The mean absolute difference in their validation probabilities for malignancy was about `1.19 × 10⁻¹⁰`, supporting the NumPy calculation. The prior-only `DummyClassifier` assigned every case the training malignant proportion, `127/341 ≈ 0.3724`. At threshold 0.50 it classified every test case as benign: accuracy was `0.631579`, but malignant recall was zero. Because it predicted no malignant cases, precision has a zero denominator; I report it as 0 using the assignment's `zero_division=0` convention. Its F1 is also 0. Thus accuracy alone hides missed malignant cases.

## Part 5: Threshold Selection and Test Results

I compared seven thresholds from 0.20 through 0.80 using **only NumPy validation probabilities**. Threshold 0.60 had the highest validation F1 (`0.831169`), so I selected it before evaluating the test set. On validation, changing 0.50 to 0.60 reduced false positives from 6 to 2 but raised false negatives from 9 to 11.

On test, the NumPy model at **0.60** had **TN = 69, FP = 3, FN = 14, TP = 28**. In plain language, it correctly identified 69 benign and 28 malignant cases, incorrectly flagged 3 benign cases, and missed 14 malignant cases. Its precision was `0.903226`, recall `0.666667`, and F1 `0.767123`. At **0.50**, precision was `0.885714`, recall `0.738095`, F1 `0.805195`, with 4 false positives and 11 false negatives. The selected threshold improved validation F1 but did not improve test F1. Both logistic models produced essentially the same results.

Changing a threshold changes class labels, **not predicted probabilities**; probability-based test log loss stayed about `0.255657`. Missing malignant cases can have serious consequences. This classroom result is not ready for clinical use: it uses only two features, has a limited sample, and lacks external validation of performance and probability calibration.

## Research Process and Sources

I initially placed the training-loss plot before `loss_history` had been created. The resulting `NameError` showed the order was wrong; I moved plotting after the training loop, reran the script, and checked that the PNG opened and showed decreasing loss. I also saw an empty CSV before the script wrote its results. I added the CSV-writing step, reran the script, and checked the rows against the printed threshold and comparison tables. Finally, I made report paths relative to the script and confirmed the required files appeared in `reports/`.

The dataset comes through scikit-learn from the [UCI source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). I used scikit-learn for data loading, splitting, scaling, the reference and baseline models, and evaluation metrics; the NumPy logistic-regression training loop is shown in `logistic_regression_analysis.py`.
