# Breast Mass Classification with Logistic Regression

## Project question

Can two measurements of a breast mass help a logistic regression model distinguish malignant from benign cases? This educational project builds logistic regression with NumPy, compares it with scikit-learn and a simple baseline, and examines how the probability threshold changes classification results. **This project is not a clinical diagnostic tool.**

## Data and approach

The script loads the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn. Its original source is the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). No separate dataset download is needed.

The dataset has 569 observations. This project uses only two features, in this order: `mean radius` and `mean texture`. The script recodes the target so **1 = malignant** (the positive class) and **0 = benign**. There are 212 malignant and 357 benign observations.

Using `random_state=42` and stratification, the script creates training (341 observations), validation (114), and test (114) sets. It fits `StandardScaler` on the training set only, then uses that fitted scaler on validation and test data. The test set is kept separate until the threshold has been selected using validation results.

The three comparison methods are:

- **NumPy logistic regression:** implemented in this project using an intercept, a numerically stable sigmoid and binary cross-entropy, and full-batch gradient descent.
- **scikit-learn logistic regression:** fitted without regularization on the same training data for a reference comparison.
- **Prior-only baseline:** a `DummyClassifier` that predicts the training-set malignant proportion as the probability for every observation.

The NumPy model starts with all coefficients equal to zero and uses a learning rate of `0.1` for `20,000` updates. These are the project's training settings, not scikit-learn defaults.

## Main findings

The NumPy model's training binary cross-entropy decreased from approximately `0.693147` to `0.246834`. Its learned coefficients, in standardized feature units, were approximately `-0.7272` for the intercept, `3.6568` for mean radius, and `1.0099` for mean texture. The scikit-learn reference produced nearly identical coefficients and validation probabilities.

Among seven validation thresholds from `0.20` to `0.80`, threshold `0.60` had the highest validation F1 score (`0.831169`). On the held-out test set, however, F1 was approximately `0.767123` at `0.60`, compared with `0.805195` at the default threshold `0.50`. At `0.60`, the NumPy model had 14 false negatives, meaning 14 malignant cases were classified as benign. The selected threshold therefore improved validation F1 but did not improve test F1. The prior-only baseline classified every test observation as benign at threshold `0.50`; its test accuracy was approximately `0.631579`, but its malignant recall was zero.

These results describe this dataset and this two-feature experiment. They do not establish clinical safety, causation, or performance on patients in other settings. See [memo.md](memo.md) for the interpretation and limitations.

## Files in this repository

- `logistic_regression_analysis.py` — loads data, trains models, evaluates thresholds, and generates results.
- `requirements.txt` — Python packages to install.
- `memo.md` — calculations, interpretation, comparisons, and limitations.
- `reports/training_loss.png` — NumPy model's training-loss plot.
- `reports/threshold_results.csv` — validation results for all seven thresholds.
- `reports/assignment_results.csv` — validation and test results for the required model and threshold configurations.

## Before you begin

You need **Python 3.13** and **Git**. A terminal is a window where you type commands and press Enter. A Git repository is a folder whose files and changes can be tracked; `git clone` downloads a copy of that folder from GitHub. You do **not** need VS Code.

1. Install Python from [python.org/downloads](https://www.python.org/downloads/). On Windows, select **“Add python.exe to PATH”** if the installer offers it. Then close and reopen PowerShell or Terminal.
2. Install Git:
   - **Windows:** download Git from the [official Windows installation page](https://git-scm.com/install/windows), run the installer, and then reopen PowerShell.
   - **macOS:** open Terminal and run `xcode-select --install`; follow the prompts to install Apple's command-line tools, which include Git. The [official macOS Git installation page](https://git-scm.com/install/mac) lists other installation methods.
3. Check the installations using the commands for your operating system below. A version number confirms the command is available.

### Windows: open PowerShell and check installations

Open the **Start** menu, type **PowerShell**, and open **Windows PowerShell** or **PowerShell**. Type:

```powershell
py -3.13 --version
git --version
```

If `py -3.13` is unavailable after installing Python, try `py --version` and confirm it reports Python 3.13. If Git is unavailable, close PowerShell, reopen it, and try again.

### macOS: open Terminal and check installations

Press **Command + Space**, type **Terminal**, and press Enter. Type:

```bash
python3 --version
git --version
```

Confirm that `python3 --version` reports Python 3.13. If it reports a different version, install Python 3.13 from python.org and use its `python3.13` command in the environment-creation step below.

## Download and run on Windows PowerShell

Run these commands **in PowerShell**. The example saves the repository in your `Downloads` folder. Each command should be entered separately.

```powershell
Set-Location "$HOME\Downloads"
git clone https://github.com/Nuquynhchaunguyen/breast-mass-classification-logistic-regression.git
Set-Location .\breast-mass-classification-logistic-regression
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python logistic_regression_analysis.py
```

`Set-Location` changes the current folder. `git clone` downloads the project. `venv` creates an isolated Python environment in `.venv`; activation makes `python` refer to that environment. `pip` installs the packages listed in `requirements.txt`. The last command runs the analysis from the project folder.

If PowerShell blocks `Activate.ps1`, **do not change your permanent execution policy**. From the project folder, use the environment's Python directly instead:

```powershell
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe .\logistic_regression_analysis.py
```

To open the generated results folder in File Explorer:

```powershell
Invoke-Item .\reports
```

## Download and run on macOS Terminal

Run these commands **in Terminal**, one at a time. The example saves the repository in your `Downloads` folder:

```bash
cd ~/Downloads
git clone https://github.com/Nuquynhchaunguyen/breast-mass-classification-logistic-regression.git
cd breast-mass-classification-logistic-regression
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python logistic_regression_analysis.py
```

If `python3 --version` showed a version other than 3.13 but you installed Python 3.13, replace only the environment-creation command with `python3.13 -m venv .venv`. Once the environment is activated, continue with `python`.

To open the generated results folder in Finder:

```bash
open reports
```

## Check that the run succeeded

The terminal should print dataset and class counts, train/validation/test counts, initial and final training losses, learned coefficients, a validation threshold table, and the selected threshold (`0.6`). Near the end, it should print that it saved both CSV files. Open `reports/` to see the three output files named above.

The script took approximately **19 seconds on the author's Mac** in one local run. Your computer may take more or less time, especially on the first run. Small differences in the last decimal places can occur with different package versions; the broad results and file names should agree.

The author's tested environment was Python `3.13.9`, NumPy `2.5.4`, Matplotlib `3.11.2`, and scikit-learn `1.9.1`. `requirements.txt` lists the required packages and allows compatible versions. To see the versions installed in your own environment, run `python -m pip list` after activation. If you used the Windows direct-Python method, run `& .\.venv\Scripts\python.exe -m pip list` instead.

To record exactly which version of this repository you ran, enter these commands from the project folder:

```text
git branch --show-current
git rev-parse --short HEAD
```

Enter them separately in PowerShell or Terminal. The first prints the branch; the second prints the commit identifier.

## Troubleshooting

- **`git` is not recognized:** install Git using the link above, close and reopen the terminal, then check `git --version`.
- **Python is not recognized or the wrong version appears:** install Python 3.13, reopen the terminal, and repeat the version check for your operating system. On Windows, use `py -3.13`; on macOS, use `python3.13` to create the environment if needed.
- **`No module named numpy`, `matplotlib`, or `sklearn`:** install dependencies in the same `.venv` used to run the script. Repeat the `pip install -r requirements.txt` command from your operating-system section.
- **`requirements.txt` or the script cannot be found:** you are probably in the wrong folder. Use `Get-Location` in PowerShell or `pwd` in macOS Terminal. Then change into `breast-mass-classification-logistic-regression` and run the command again.
- **PowerShell blocks environment activation:** use the three direct `.\.venv\Scripts\python.exe` commands shown in the Windows section.
- **`git clone` says the destination folder already exists:** you already have a folder with that name in the current location. Enter that folder if it is the correct project, or choose another download location. Do not delete an existing folder without checking its contents.
- **Output files do not appear:** run the script from the project folder, the one containing `requirements.txt` and `logistic_regression_analysis.py`. Check the terminal for an error before the “Saved” messages.
