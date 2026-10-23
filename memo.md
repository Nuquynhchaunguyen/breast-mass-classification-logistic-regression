# Breast Mass Classification: Logistic Regression and Decision Thresholds

## Part 1: From Scores to Decisions

The one-feature Explorer is separate from the dataset below. Its score is `z = β₀ + β₁x`, its probability is `p = 1 / (1 + exp(-z))`, and its predicted class is 1 when `p ≥ threshold`.

| Case | β₀ | β₁ | x | Threshold | Score z | Probability p | Predicted class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

For A, `z = -1.0 + 0.8 × 2.0 = 0.60`, so `p = 1 / (1 + exp(-0.60)) ≈ 0.6457`, above 0.50. B keeps that probability but raises the threshold to 0.70, changing the label to 0. In C, the negative coefficient makes probability decrease as `x` increases. In D, the zero coefficient makes probability constant. These Explorer coefficients were set manually; the NumPy model learns its coefficients from data.

## Part 2: Data Preparation

I used scikit-learn's bundled Breast Cancer Wisconsin (Diagnostic) dataset, originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). Its measurements describe cell nuclei, not patient demographics. I selected only `mean radius` and `mean texture`, in that order. Neither feature has missing values among 569 observations. I recoded the target to **1 = malignant** (positive) and **0 = benign**: 212 malignant and 357 benign cases.

| Split | Total | Benign (0) | Malignant (1) |
|---|---:|---:|---:|
| Training | 341 | 214 | 127 |
| Validation | 114 | 71 | 43 |
| Test | 114 | 72 | 42 |

I split row indices into development and test sets with `test_size=0.20`, `stratify=y`, and `random_state=42`, then split development into training and validation with `test_size=0.25`, stratified development labels, and the same seed. Thus validation is 25% of the remaining 80%, yielding about 60% training, 20% validation, and 20% test. No rows overlap; all 569 are assigned. I fitted `StandardScaler` only on training rows and reused it on validation and test rows to prevent preprocessing leakage.

## Part 3: NumPy Logistic Regression

My NumPy model uses an intercept and two standardized features. All coefficients began at zero. Full-batch gradient descent used learning rate `0.1` for `20,000` updates, without regularization or class weights. Minimizing mean binary cross-entropy maximizes observed-label likelihood. Stable sigmoid and loss calculations avoided overflow.

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

I compared seven thresholds from 0.20 through 0.80 using **only NumPy validation probabilities**, choosing the highest F1 and the smallest threshold in a tie. Threshold 0.60 won (`0.831169`) before test evaluation. On validation, moving from 0.50 to 0.60 reduced false positives from 6 to 2 but raised false negatives from 9 to 11.

On test, the NumPy model at **0.60** produced this confusion matrix (rows = actual; columns = predicted):

| Actual / Predicted | Benign (0) | Malignant (1) |
|---|---:|---:|
| Benign (0) | TN = 69 | FP = 3 |
| Malignant (1) | FN = 14 | TP = 28 |

It correctly identified 69 benign and 28 malignant cases, falsely flagged 3 benign cases, and missed 14 malignant cases. Precision was `0.903226`, recall `0.666667`, and F1 `0.767123`. At **0.50**, precision was `0.885714`, recall `0.738095`, and F1 `0.805195`, with 4 false positives and 11 false negatives. The selected threshold improved validation F1 but not test F1. Both logistic models gave essentially identical results.

Changing a threshold changes class labels, **not predicted probabilities**; probability-based test log loss stayed about `0.255657`. Missing malignant cases can have serious consequences. This classroom result is not ready for clinical use: it uses only two features, has a limited sample, and lacks external validation of performance and probability calibration.

## Research Process and Sources

I initially placed the training-loss plot before `loss_history` had been created. The resulting `NameError` showed the order was wrong; I moved plotting after the training loop, reran the script, and checked that the PNG opened and showed decreasing loss. I also saw an empty CSV before the script wrote its results. I added the CSV-writing step, reran the script, and checked the rows against the printed threshold and comparison tables. Finally, I made report paths relative to the script and confirmed the required files appeared in `reports/`.

The dataset comes through scikit-learn from the [UCI source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). I used scikit-learn for data loading, splitting, scaling, the reference and baseline models, and evaluation metrics; the NumPy logistic-regression training loop is shown in `logistic_regression_analysis.py`.
