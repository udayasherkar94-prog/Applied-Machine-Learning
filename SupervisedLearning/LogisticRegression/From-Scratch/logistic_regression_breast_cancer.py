import copy
import math
import numpy as np


# =========================================================
# 1. SAMPLE BREAST TUMOR DATA
# =========================================================
#
# We are creating a small dataset ourselves.
#
# Features:
#
# x0 = tumor size in cm
# x1 = cell density score
#
# Target:
#
# 0 = Malignant
# 1 = Benign
#
# IMPORTANT:
# This is artificial data created only for learning.
# It is NOT real medical data.
# =========================================================

X_train = np.array([
    [2.0, 1.0],
    [2.5, 1.5],
    [3.0, 1.2],
    [3.5, 2.0],
    [4.0, 2.5],

    [6.0, 6.0],
    [6.5, 7.0],
    [7.0, 6.5],
    [7.5, 8.0],
    [8.0, 7.5]
])


# 0 = Malignant
# 1 = Benign

y_train = np.array([
    0,
    0,
    0,
    0,
    0,

    1,
    1,
    1,
    1,
    1
])


print("Training data:")
print(X_train)

print("\nActual target:")
print(y_train)


# =========================================================
# 2. FEATURE SCALING FROM SCRATCH
# =========================================================
#
# We will NOT use StandardScaler from sklearn.
#
# Formula:
#
# z = (x - mean) / standard_deviation
#
# We calculate mean and standard deviation ourselves.
# =========================================================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)


# Standardize the training data
X_scaled = (X_train - mean) / std


print("\nMean:")
print(mean)

print("\nStandard deviation:")
print(std)

print("\nScaled X:")
print(X_scaled)


# =========================================================
# 3. SIGMOID FUNCTION
# =========================================================
#
# Converts z into a probability between 0 and 1.
#
# sigmoid(z) = 1 / (1 + e^(-z))
# =========================================================

def sigmoid(z):

    return 1 / (1 + np.exp(-z))


# =========================================================
# 4. LOGISTIC COST FUNCTION
# =========================================================

def compute_cost_logistic(X, y, w, b):

    # Number of training examples
    m = X.shape[0]

    cost = 0

    # Go through every training example
    for i in range(m):

        # Calculate:
        #
        # z = w.x + b
        #
        z = np.dot(X[i], w) + b

        # Convert z into probability
        f_wb = sigmoid(z)

        # Prevent log(0)
        f_wb = np.clip(
            f_wb,
            1e-15,
            1 - 1e-15
        )

        # Logistic loss
        loss = (
            -y[i] * np.log(f_wb)
            - (1 - y[i]) * np.log(1 - f_wb)
        )

        # Add loss
        cost = cost + loss

    # Average cost
    cost = cost / m

    return cost


# =========================================================
# 5. COMPUTE GRADIENT
# =========================================================

def compute_gradient_logistic(X, y, w, b):

    # m = number of examples
    # n = number of features
    m, n = X.shape

    # Gradient for every weight
    #
    # For 2 features:
    #
    # dj_dw = [0, 0]
    #
    dj_dw = np.zeros(n)

    # Gradient for bias
    dj_db = 0.0


    # -----------------------------------------------------
    # Process every training example
    # -----------------------------------------------------

    for i in range(m):

        # Calculate z
        z = np.dot(X[i], w) + b

        # Calculate prediction
        f_wb_i = sigmoid(z)

        # Calculate error
        #
        # error = prediction - actual
        #
        error = f_wb_i - y[i]


        # -------------------------------------------------
        # Calculate gradient for every feature
        # -------------------------------------------------

        for j in range(n):

            dj_dw[j] = (
                dj_dw[j]
                + error * X[i, j]
            )


        # Gradient for b
        dj_db = dj_db + error


    # Average gradient
    dj_dw = dj_dw / m

    dj_db = dj_db / m


    return dj_db, dj_dw


# =========================================================
# 6. GRADIENT DESCENT
# =========================================================

def gradient_descent(
    X,
    y,
    w_in,
    b_in,
    alpha,
    num_iters
):

    # Store cost after every iteration
    J_history = []

    # Copy initial weights
    w = copy.deepcopy(w_in)

    # Initial bias
    b = b_in


    # -----------------------------------------------------
    # Repeat gradient descent
    # -----------------------------------------------------

    for i in range(num_iters):

        # Calculate gradient
        dj_db, dj_dw = compute_gradient_logistic(
            X,
            y,
            w,
            b
        )


        # Update weights
        #
        # w = w - alpha * gradient
        #
        w = w - alpha * dj_dw


        # Update bias
        b = b - alpha * dj_db


        # Calculate cost
        cost = compute_cost_logistic(
            X,
            y,
            w,
            b
        )


        # Store cost
        J_history.append(cost)


        # Print cost every 100 iterations
        if i % 100 == 0:

            print(
                f"Iteration {i:4d}: "
                f"Cost = {cost:.6f}"
            )


    return w, b, J_history


# =========================================================
# 7. INITIALIZE w AND b
# =========================================================

# Number of features
n_features = X_scaled.shape[1]


# Initially all weights are zero
#
# Because we have 2 features:
#
# w = [0, 0]
#
w_initial = np.zeros(n_features)


# Initially bias is zero
b_initial = 0.0


# Learning rate
alpha = 0.1


# Number of iterations
num_iters = 2000


print("\nStarting Gradient Descent...\n")


# =========================================================
# 8. TRAIN THE MODEL
# =========================================================

w_final, b_final, J_history = gradient_descent(
    X_scaled,
    y_train,
    w_initial,
    b_initial,
    alpha,
    num_iters
)


print("\nTraining completed.")


print("\nFinal weights:")
print(w_final)


print("\nFinal bias:")
print(b_final)


# =========================================================
# 9. PREDICTION FUNCTION
# =========================================================

def predict(X, w, b):

    # Calculate z
    z = np.dot(X, w) + b

    # Convert z into probability
    probability = sigmoid(z)


    # Convert probability into class
    #
    # probability >= 0.5 -> 1
    # probability < 0.5  -> 0

    if probability >= 0.5:
        prediction = 1
    else:
        prediction = 0


    return prediction, probability


# =========================================================
# 10. TEST OUR EXISTING TRAINING DATA
# =========================================================

print("\n========================================")
print("TRAINING DATA PREDICTIONS")
print("========================================")


for i in range(len(X_scaled)):

    prediction, probability = predict(
        X_scaled[i],
        w_final,
        b_final
    )


    print(
        f"Patient {i + 1}: "
        f"Actual = {y_train[i]}, "
        f"Predicted = {prediction}, "
        f"Probability = {probability:.4f}"
    )


# =========================================================
# 11. TEST A NEW TUMOR
# =========================================================
#
# This tumor was NOT used during training.
#
# Suppose a new patient has:
#
# tumor size = 3.2 cm
# cell density = 1.8
# =========================================================

new_tumor = np.array([
    [3.2, 1.8]
])


# IMPORTANT:
#
# We must scale the new data using the SAME
# mean and standard deviation calculated
# from the training data.
#
new_tumor_scaled = (
    new_tumor - mean
) / std


# Make prediction
prediction, probability = predict(
    new_tumor_scaled[0],
    w_final,
    b_final
)


print("\n========================================")
print("NEW TUMOR PREDICTION")
print("========================================")


print("Tumor size:", new_tumor[0][0], "cm")

print("Cell density:", new_tumor[0][1])


print("\nProbability:", probability)


print("Predicted class:", prediction)


if prediction == 0:

    print("Prediction: Malignant")

else:

    print("Prediction: Benign")