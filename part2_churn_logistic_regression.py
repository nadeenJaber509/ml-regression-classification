import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load data + basic cleaning
df = pd.read_csv("customer_churn_data.csv")
df = df.drop("CustomerID", axis=1)

# Fill missing numeric values with median 
for col in ['Age', 'Income', 'SupportCalls', 'Tenure']:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("ChurnStatus", axis=1)
y = df["ChurnStatus"]

# Standardize features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  Manual split 2500 train / 500 val / 500 test
# Using stratify to keep class balance
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=500, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=500, random_state=42, stratify=y_temp
)


# returns metrics in table format
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}

    # Predictions
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # Save metrics
    results['Train'] = [
        accuracy_score(y_train, pred_train),
        precision_score(y_train, pred_train, zero_division=0),
        recall_score(y_train, pred_train, zero_division=0)
    ]
    results['Validation'] = [
        accuracy_score(y_val, pred_val),
        precision_score(y_val, pred_val, zero_division=0),
        recall_score(y_val, pred_val, zero_division=0)
    ]
    results['Test'] = [
        accuracy_score(y_test, pred_test),
        precision_score(y_test, pred_test, zero_division=0),
        recall_score(y_test, pred_test, zero_division=0)
    ]
    return results

# Function to display a table for any model
def print_table(title, results):
    df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall"])
    print("\n", title)
    print(df, "\n")


# Linear Logistic Regression
linear_model = LogisticRegression(max_iter=2000, class_weight="balanced")
linear_model.fit(X_train, y_train)

linear_results = evaluate_model(
    linear_model, X_train, y_train, X_val, y_val, X_test, y_test
)

print_table("Linear Logistic Regression", linear_results)


# Polynomial Logistic Regression
poly_degrees = [2, 5, 9]
poly_results = {}
poly_models = {}
poly_transforms = {}

for d in poly_degrees:
    # Polynomial transformation (degree controls complexity)
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_val_p = poly.transform(X_val)
    X_test_p = poly.transform(X_test)

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train_p, y_train)

    poly_models[d] = model
    poly_transforms[d] = (X_train_p, X_val_p, X_test_p)

    res = evaluate_model(model, X_train_p, y_train, X_val_p, y_val, X_test_p, y_test)
    poly_results[d] = res

    print_table(f"Polynomial Logistic Regression (degree {d})", res)


# Select best model using validation F1 score
# (precision+recall balance)
best_name = None
best_score = -1

for name, res in {"Linear": linear_results,
                  "Poly-2": poly_results[2],
                  "Poly-5": poly_results[5],
                  "Poly-9": poly_results[9]}.items():

    p = res["Validation"][1]
    r = res["Validation"][2]
    f1 = 0 if (p+r)==0 else 2*p*r/(p+r)

    if f1 > best_score:
        best_score = f1
        best_name = name

print("\nBest model based on Validation F1:", best_name)


# ROC curve for best model only
if best_name == "Linear":
    best_model = linear_model
    Xv = X_val
    Xt = X_test
else:
    deg = int(best_name.split("-")[1])
    best_model = poly_models[deg]
    Xv = poly_transforms[deg][1]
    Xt = poly_transforms[deg][2]

# ROC curve values
val_scores = best_model.predict_proba(Xv)[:, 1]
test_scores = best_model.predict_proba(Xt)[:, 1]

fpr_v, tpr_v, _ = roc_curve(y_val, val_scores)
fpr_t, tpr_t, _ = roc_curve(y_test, test_scores)
 
auc_v = auc(fpr_v, tpr_v)
auc_t = auc(fpr_t, tpr_t)

print("\nValidation AUC:", auc_v)
print("Test AUC:", auc_t)

# Plot ROC
plt.figure(figsize=(8,5))
plt.plot(fpr_v, tpr_v, label=f"Validation AUC={auc_v:.3f}")
plt.plot(fpr_t, tpr_t, label=f"Test AUC={auc_t:.3f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve – {best_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
