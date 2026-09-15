import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def load_house_data():
    """
    Load the house price dataset.

    Features:
    1. Size in sqft
    2. Number of bedrooms
    3. Number of floors
    4. Age of house

    Target:
    House price in thousands of dollars.
    """

    X = np.array([
        [952, 2, 1, 65],
        [1244, 3, 2, 64],
        [1947, 3, 2, 17],
        [1722, 4, 2, 42],
        [1218, 3, 2, 15],
        [1656, 3, 1, 18],
        [1219, 3, 2, 21],
        [1600, 3, 2, 19],
        [1200, 3, 1, 40],
        [1800, 4, 2, 15]
    ])

    y = np.array([
        271.5,
        232.0,
        509.8,
        477.0,
        218.0,
        400.0,
        300.0,
        420.0,
        318.7,
        450.0
    ])

    return X, y


# ---------------------------------------------------------
# Compute Cost
# ---------------------------------------------------------

def compute_cost(X, y, w, b):
    """
    Compute the cost function for multiple linear regression.
    """

    m = X.shape[0]

    predictions = np.dot(X, w) + b

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


# ---------------------------------------------------------
# Compute Gradient
# ---------------------------------------------------------

def compute_gradient(X, y, w, b):
    """
    Compute gradients for weights and bias.
    """

    m = X.shape[0]

    predictions = np.dot(X, w) + b

    errors = predictions - y

    dj_dw = (1 / m) * np.dot(X.T, errors)

    dj_db = (1 / m) * np.sum(errors)

    return dj_dw, dj_db


# ---------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------

def gradient_descent(X, y, w, b, alpha, iterations):
    """
    Perform gradient descent to minimize the cost function.
    """

    cost_history = []

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)

        cost_history.append(cost)

        if i % 100 == 0:
            print(
                f"Iteration {i}: "
                f"Cost = {cost:.2f}"
            )

    return w, b, cost_history


# ---------------------------------------------------------
# Z-Score Normalization
# ---------------------------------------------------------

def zscore_normalize_features(X):
    """
    Normalize features using Z-score normalization.

    Formula:

        X_norm = (X - mean) / standard_deviation

    Returns:
        X_norm
        mean
        standard_deviation
    """

    mu = np.mean(X, axis=0)

    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

X_train, y_train = load_house_data()

print("Original Features:")
print(X_train)

print("\nTarget Values:")
print(y_train)


# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------

X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

print("\nMean of Features:")
print(X_mu)

print("\nStandard Deviation:")
print(X_sigma)

print("\nNormalized Features:")
print(X_norm)


# ---------------------------------------------------------
# Initialize Parameters
# ---------------------------------------------------------

n_features = X_norm.shape[1]

w = np.zeros(n_features)

b = 0

alpha = 0.1

iterations = 1000


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

w, b, cost_history = gradient_descent(
    X_norm,
    y_train,
    w,
    b,
    alpha,
    iterations
)


# ---------------------------------------------------------
# Display Final Parameters
# ---------------------------------------------------------

print("\nFinal Parameters:")

print("Weights:", w)

print("Bias:", b)

print("Final Cost:", cost_history[-1])


# ---------------------------------------------------------
# Cost Graph
# ---------------------------------------------------------

plt.plot(cost_history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.title("Cost vs Iterations")

plt.show()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

x_house = np.array([
    1200,
    3,
    1,
    40
])


# Normalize new input
x_house_norm = (x_house - X_mu) / X_sigma


# Predict
prediction = np.dot(x_house_norm, w) + b


print("\nNew House Information:")

print("Size: 1200 sqft")

print("Bedrooms: 3")

print("Floors: 1")

print("Age: 40 years")


print(
    f"\nPredicted House Price: "
    f"${prediction * 1000:,.0f}"
)