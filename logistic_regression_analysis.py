"""Prepare data and compare two-feature logistic regression models.

Assignment map: Part 1's four manual score examples are in memo.md.
Parts 2-6 are implemented below and marked where each part begins.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Part 1: The four manual score examples and explanations are in memo.md.
# Part 2: Prepare a leakage-free experiment.
# Find the reports folder relative to this script, not the terminal's folder.
PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# 1. Load and inspect the dataset.
# The original scikit-learn target is 0 = malignant, 1 = benign.
data = load_breast_cancer()

print("Number of samples:", data.data.shape[0])
print("Available features:", data.feature_names)
print("Original target names:", data.target_names)


# 2. Keep only the two features required for this project.
# Looking up features by name makes their order explicit.
feature_names = ["mean radius", "mean texture"]
feature_indices = [
    list(data.feature_names).index(name) for name in feature_names
]
X = data.data[:, feature_indices]

# Recode the target so a positive prediction means malignant:
# 1 = malignant, 0 = benign.
y = (data.target == 0).astype(int)

print("Selected features:", feature_names)
print("X shape:", X.shape)
print("Missing values:", np.isnan(X).sum())
print("Benign (0):", np.sum(y == 0))
print("Malignant (1):", np.sum(y == 1))


# 3. Split row indices into train, validation, and test sets.
# First reserve 20% for test. Then take 25% of the remaining 80%
# for validation, giving approximately 60% train / 20% val / 20% test.
# Stratification keeps class proportions similar in each split.
all_indices = np.arange(len(y))

dev_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

train_indices, val_indices = train_test_split(
    dev_indices,
    test_size=0.25,
    stratify=y[dev_indices],
    random_state=42,
)

for name, indices in [
    ("Train", train_indices),
    ("Validation", val_indices),
    ("Test", test_indices),
]:
    labels, counts = np.unique(y[indices], return_counts=True)
    class_counts = dict(zip(labels, counts))
    print(
        f"{name}: {len(indices)} samples; "
        f"benign (0) = {class_counts.get(0, 0)}, "
        f"malignant (1) = {class_counts.get(1, 0)}"
    )


# 4. Verify that every row belongs to exactly one split.
train_set = set(train_indices)
val_set = set(val_indices)
test_set = set(test_indices)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)
assert train_set | val_set | test_set == set(all_indices)

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]


# 5. Standardize features without leaking validation or test information.
# The scaler learns its mean and standard deviation from training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Split check: no overlap; all 569 samples accounted for")
print("Training mean:", scaler.mean_)
print("Training scale:", scaler.scale_)
print("Scaled training mean:", X_train_scaled.mean(axis=0))


# Part 3: Implement NumPy logistic regression.
# 6. Add an intercept column to the training matrix.
# The other two columns are standardized radius and texture.
X_train_aug = np.column_stack(
    [np.ones(len(X_train_scaled)), X_train_scaled]
)


# 7. Define the logistic probability function.
# logaddexp avoids overflow for large positive or negative scores.
def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


# 8. Define mean binary cross-entropy.
# For a score z, logaddexp(0, z) - y*z is a stable loss calculation.
def binary_cross_entropy(X_aug, y_true, coefficients):
    scores = X_aug @ coefficients
    return np.mean(np.logaddexp(0, scores) - y_true * scores)


# 9. Start all coefficients at zero and measure the initial loss.
coefficients = np.zeros(X_train_aug.shape[1])
initial_loss = binary_cross_entropy(X_train_aug, y_train, coefficients)

print("Initial coefficients:", coefficients)
print("Initial training loss:", initial_loss)


# 10. Learn coefficients using full-batch gradient descent.
# Each update uses every training observation. Validation and test data
# do not participate in fitting the coefficients.
learning_rate = 0.1
epochs = 20_000
loss_history = [initial_loss]

for _ in range(epochs):
    scores = X_train_aug @ coefficients
    probabilities = sigmoid(scores)

    # Gradient of mean binary cross-entropy for each coefficient.
    gradient = X_train_aug.T @ (probabilities - y_train) / len(y_train)
    coefficients -= learning_rate * gradient

    # Record loss after the update for the training-loss plot.
    loss_history.append(
        binary_cross_entropy(X_train_aug, y_train, coefficients)
    )

print("Final training loss:", loss_history[-1])
print("Learned intercept:", coefficients[0])
print("Learned mean radius coefficient:", coefficients[1])
print("Learned mean texture coefficient:", coefficients[2])
print("Feature coefficients use standardized training units.")
print("Loss decreased:", loss_history[-1] < loss_history[0])


# 11. Save a plot showing how training loss changed.
plt.figure(figsize=(8, 5))
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel("Gradient descent update")
plt.ylabel("Training binary cross-entropy")
plt.title("NumPy Logistic Regression Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "training_loss.png", dpi=160)
plt.close()

print("Saved reports/training_loss.png")


# Part 4: Compare with scikit-learn and a prior-only baseline.
# 12. Fit scikit-learn logistic regression on the same training data.
# C=np.inf disables regularization for comparison with the NumPy model.
sklearn_model = LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10_000,
    tol=1e-9,
)
sklearn_model.fit(X_train_scaled, y_train)

# A prior-only baseline predicts using the training class proportions.
baseline_model = DummyClassifier(strategy="prior")
baseline_model.fit(X_train_scaled, y_train)

# Both probability arrays must refer to class 1 (malignant).
assert list(sklearn_model.classes_) == [0, 1]
assert list(baseline_model.classes_) == [0, 1]

numpy_val_prob = sigmoid(
    coefficients[0] + X_val_scaled @ coefficients[1:]
)
sklearn_val_prob = sklearn_model.predict_proba(X_val_scaled)[:, 1]
baseline_val_prob = baseline_model.predict_proba(X_val_scaled)[:, 1]

mean_probability_difference = np.mean(
    np.abs(numpy_val_prob - sklearn_val_prob)
)

print("NumPy coefficients:", coefficients)
print(
    "scikit-learn coefficients:",
    np.r_[sklearn_model.intercept_, sklearn_model.coef_.ravel()],
)
print(
    "Mean absolute validation probability difference:",
    mean_probability_difference,
)
print("Baseline malignant probability:", baseline_val_prob[0])


# Part 5: Select a threshold using validation data, then evaluate.
# 13. Choose a threshold using NumPy validation probabilities only.
# Keep the test set untouched while making this decision.
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
threshold_results = []

for threshold in thresholds:
    predictions = (numpy_val_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()
    f1 = f1_score(y_val, predictions, zero_division=0)

    threshold_results.append((threshold, tn, fp, fn, tp, f1))

print("Validation threshold results:")
print("threshold   TN  FP  FN  TP      F1")
for threshold, tn, fp, fn, tp, f1 in threshold_results:
    print(f"{threshold:8.2f}  {tn:3d} {fp:3d} {fn:3d} {tp:3d}  {f1:.6f}")

# Python's max keeps the first result in a tie; thresholds are ascending.
selected_threshold = max(
    threshold_results, key=lambda row: row[-1]
)[0]
print("Selected threshold:", selected_threshold)


# Part 6: Export the seven validation-threshold results.
# 14. Save the validation threshold sweep.
with open(
    REPORTS_DIR / "threshold_results.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "threshold", "tn", "fp", "fn", "tp",
            "accuracy", "precision", "recall", "f1",
        ]
    )

    for threshold, tn, fp, fn, tp, f1 in threshold_results:
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        writer.writerow([
            f"{threshold:.2f}", tn, fp, fn, tp,
            f"{accuracy:.6f}",
            f"{precision:.6f}",
            f"{recall:.6f}",
            f"{f1:.6f}",
        ])

print("Saved reports/threshold_results.csv")


# Part 5 continued: Evaluate the frozen models and selected threshold.
# 15. Calculate confusion counts and evaluation metrics.
# Class 1 is malignant, so FN means a missed malignant case.
def evaluate_predictions(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    # Log loss uses probabilities, not thresholded class labels.
    probability_log_loss = log_loss(
        y_true, probabilities, labels=[0, 1]
    )

    return (
        tn, fp, fn, tp,
        accuracy, precision, recall, f1, probability_log_loss,
    )


# 16. Evaluate five configurations on validation and test data.
# The test labels are used only for final reporting.
assignment_results = []

for split_name, scaled_features, true_labels in [
    ("validation", X_val_scaled, y_val),
    ("test", X_test_scaled, y_test),
]:
    numpy_prob = sigmoid(
        coefficients[0] + scaled_features @ coefficients[1:]
    )
    sklearn_prob = sklearn_model.predict_proba(scaled_features)[:, 1]
    baseline_prob = baseline_model.predict_proba(scaled_features)[:, 1]

    configurations = [
        ("baseline", "default", 0.50, baseline_prob),
        ("numpy", "default", 0.50, numpy_prob),
        ("numpy", "selected", selected_threshold, numpy_prob),
        ("sklearn", "default", 0.50, sklearn_prob),
        ("sklearn", "selected", selected_threshold, sklearn_prob),
    ]

    for model_name, policy, threshold, probabilities in configurations:
        metrics = evaluate_predictions(
            true_labels, probabilities, threshold
        )
        assignment_results.append(
            (split_name, model_name, policy, threshold, *metrics)
        )


# Part 6 continued: Print and export the final comparison results.
# 17. Print a readable comparison table in the terminal.
print("\nAssignment comparison results:")
print("TN: benign -> benign; FP: benign -> malignant; FN: malignant -> benign; TP: malignant -> malignant")
print(
    f"{'split':<11} {'model':<9} {'policy':<8} {'cutoff':>6} "
    f"{'TN':>3} {'FP':>3} {'FN':>3} {'TP':>3} "
    f"{'accuracy':>9} {'precision':>9} {'recall':>9} "
    f"{'F1':>9} {'log loss':>9}"
)
for row in assignment_results:
    split_name, model_name, policy, threshold = row[:4]
    tn, fp, fn, tp = row[4:8]
    accuracy, precision, recall, f1, probability_log_loss = row[8:]

    print(
        f"{split_name:<11} {model_name:<9} {policy:<8} {threshold:>6.2f} "
        f"{tn:>3} {fp:>3} {fn:>3} {tp:>3} "
        f"{accuracy:>9.6f} {precision:>9.6f} {recall:>9.6f} "
        f"{f1:>9.6f} {probability_log_loss:>9.6f}"
    )


# 18. Save the same ten configurations to a CSV file.
with open(
    REPORTS_DIR / "assignment_results.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "split", "model", "policy", "threshold",
        "tn", "fp", "fn", "tp",
        "accuracy", "precision", "recall", "f1", "log_loss",
    ])

    for row in assignment_results:
        split_name, model_name, policy, threshold = row[:4]
        tn, fp, fn, tp = row[4:8]
        accuracy, precision, recall, f1, probability_log_loss = row[8:]

        writer.writerow([
            split_name, model_name, policy, f"{threshold:.2f}",
            tn, fp, fn, tp,
            f"{accuracy:.6f}",
            f"{precision:.6f}",
            f"{recall:.6f}",
            f"{f1:.6f}",
            f"{probability_log_loss:.6f}",
        ])

print("Saved reports/assignment_results.csv")
