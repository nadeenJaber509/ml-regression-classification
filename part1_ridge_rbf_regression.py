import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Fixed random values
np.random.seed(42)

# Create 25 evenly spaced x-values between 0 and 1
x = np.linspace(0, 1, 25)

# Generate random noise uniformly in [-0.3, 0.3]
noise = np.random.uniform(-0.3, 0.3, size=25)

# Build the target function y = sin(5πx) with added noise
y = np.sin(5 * np.pi * x) + noise

for i in range(25):
    print(f"x[{i}] = {x[i]:.4f},  y[{i}] = {y[i]:.4f}")



# PART ONE - A
# We're using polynomial features up to degree 9: 1, x, x^2, ..., x^9 (10 features) 
degree = 9

# design polynomial matrix 
X = np.vstack([x**i for i in range(degree + 1)]).T


# Test several lambda values for ridge regression
lambdas = [0, 0.0001, 0.01, 1, 10]

ridge_models = {}   # Will store the learned parameters (theta) for each lambda

# Compute the ridge regression for each lambda
for lam in lambdas:
    # Identity matrix for regularization
    I = np.eye(X.shape[1])
    
    # Closed-form ridge solution: (XᵀX + λI)⁻¹ Xᵀ y
    theta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    
    ridge_models[lam] = theta


# Create a dense set of points so the predicted curves look smooth
x_plot = np.linspace(0, 1, 300)

# Build polynomial features for the smoother curve
X_plot = np.vstack([x_plot**i for i in range(degree + 1)]).T

plt.figure(figsize=(10, 6))

# Plot the original noisy data
plt.scatter(x, y, color='black', label='Noisy Data', s=40)

# Plot the ridge regression predictions for each lambda
for lam in lambdas:
    theta = ridge_models[lam]
    y_pred = X_plot @ theta
    plt.plot(x_plot, y_pred, label=f"λ = {lam}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ridge Regression Models for Different λ Values")
plt.legend()
plt.grid(True)
plt.show()



# PART ONE - B
# Trying several values for the number of RBFs
K_values = [1, 5, 10, 50]
print("RBF configurations:", K_values)

# Evenly distribute RBF centers across [0,1]
def get_rbf_centers(K):
    return np.linspace(0, 1, K)

# Pick a reasonable sigma based on spacing between centers
def choose_sigma(K):
    if K == 1:
        return 0.4   
    d = 1.0 / (K - 1)   # Distance between centers
    return d * 1.5      

# Build the RBF design matrix (Gaussian basis functions)
def rbf_design_matrix(x, centers, sigma):
    N = len(x)
    K = len(centers)
    Phi = np.zeros((N, K + 1))
    
    # Bias term
    Phi[:, 0] = 1.0
    
    # Each RBF column is exp(-(x - center)² / (2σ²))
    for j in range(K):
        Phi[:, j+1] = np.exp(-((x - centers[j])**2) / (2 * sigma**2))
    return Phi

# Train an RBF model for each K value
rbf_models = {}

for K in K_values:
    centers = get_rbf_centers(K)
    sigma = choose_sigma(K)
    Phi = rbf_design_matrix(x, centers, sigma)

    # Solve using pseudoinverse (no regularization here)
    theta = np.linalg.pinv(Phi) @ y
    rbf_models[K] = {
        "theta": theta,
        "centers": centers,
        "sigma": sigma
    }
    print(f"Trained RBF model for K={K}: theta shape = {theta.shape}")

x_plot = np.linspace(0, 1, 300)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', s=40, label='Noisy Data')
plt.plot(x_plot, np.sin(5*np.pi*x_plot), 'k--', label='True Function')

# Plot the RBF model predictions for each value of K
for K in K_values:
    model = rbf_models[K]
    theta = model["theta"]
    centers = model["centers"]
    sigma = model["sigma"]

    Phi_plot = rbf_design_matrix(x_plot, centers, sigma)
    y_pred = Phi_plot @ theta

    plt.plot(x_plot, y_pred, linewidth=2, label=f"K = {K}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("RBF Regression with Different Numbers of Basis Functions")
plt.legend()
plt.grid(True)
plt.show()
