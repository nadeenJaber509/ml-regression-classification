# Regression and Classification ML Project

This project demonstrates two core machine learning tasks:

- Function approximation with Polynomial Ridge Regression and Gaussian RBF Regression.
- Binary customer churn prediction with Logistic Regression and polynomial feature expansion.

The repository is organized for clear reproducibility and easy review on GitHub.

## Highlights

- End-to-end modeling workflows in NumPy, Pandas, and scikit-learn.
- Comparison of multiple model complexities and regularization settings.
- Standardized evaluation using Accuracy, Precision, Recall, ROC, and AUC.
- Visual outputs for model behavior and classification performance.

## Repository Structure

- `part1_ridge_rbf_regression.py`: Synthetic-data regression experiments and plots.
- `part2_churn_logistic_regression.py`: Churn classification pipeline and ROC analysis.
- `customer_churn_data.csv`: Input dataset for churn prediction.
- `.gitignore`: Python and environment ignore rules.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Setup

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

From the project root:

```bash
python part1_ridge_rbf_regression.py
python part2_churn_logistic_regression.py
```

## Part 1: Regression Modeling

The first script generates noisy observations from:

$$y = \sin(5\pi x) + \epsilon, \quad \epsilon \sim U(-0.3, 0.3)$$

It then compares:

1. Degree-9 Polynomial Ridge Regression for multiple regularization values.
2. Gaussian RBF Regression for different numbers of basis functions.

Output:

- Regression curves overlaid on noisy samples.
- Visual comparison of underfitting, good fit, and overfitting behavior.

## Part 2: Churn Classification

The second script builds a customer churn classifier with data preprocessing and model selection:

1. Drops identifier column and imputes missing numeric values with median.
2. Standardizes features.
3. Splits into train, validation, and test sets with stratification.
4. Trains linear and polynomial logistic regression models.
5. Selects the best model using validation F1 score:

$$F_1 = \frac{2PR}{P + R}$$

6. Plots ROC curves and reports validation and test AUC.

## Reproducibility

- Fixed random seeds are used where applicable.
- Scripts are deterministic given the same environment and dataset.

## Author

- Student IDs: 1220439, 1220044

## Suggested GitHub Repository Metadata

For a stronger public project page, use:

- Repository name: `ml-assignment2-regression-classification`
- Description: `Regression and classification ML project using ridge, RBF, and logistic regression with ROC/AUC evaluation.`
- Topics: `machine-learning`, `python`, `scikit-learn`, `regression`, `classification`, `logistic-regression`, `ridge-regression`, `rbf`
