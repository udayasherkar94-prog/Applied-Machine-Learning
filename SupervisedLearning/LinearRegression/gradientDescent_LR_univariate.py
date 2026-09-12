import math, copy
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Training Data
# ============================================================

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# ============================================================
# 2. Compute Cost
# ============================================================

def compute_cost(x, y, w, b):
    """
    Calculate the cost for linear regression.

    x : training data
    y : target values
    w : parameter w
    b : parameter b
    """

    m = x.shape[0]

    cost = 0

    for i in range(m):

        # Calculate prediction
        f_wb = w * x[i] + b

        # Calculate squared error
        cost = cost + (f_wb - y[i])**2

    # Calculate total cost
    total_cost = 1 / (2 * m) * cost

    return total_cost


# ============================================================
# 3. Compute Gradient
# ============================================================

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.

    Returns:
        dj_dw : gradient with respect to w
        dj_db : gradient with respect to b
    """

    # Number of training examples
    m = x.shape[0]

    # Initialize gradients
    dj_dw = 0
    dj_db = 0

    # Calculate gradient for every training example
    for i in range(m):

        # Calculate prediction
        f_wb = w * x[i] + b

        # Gradient with respect to w
        dj_dw_i = (f_wb - y[i]) * x[i]

        # Gradient with respect to b
        dj_db_i = f_wb - y[i]

        # Add gradients
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    # Average the gradients
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# ============================================================
# 4. Gradient Descent
# ============================================================

def gradient_descent(
        x,
        y,
        w_in,
        b_in,
        alpha,
        num_iters,
        cost_function,
        gradient_function
    ):

    """
    Performs gradient descent to find the best w and b.
    """

    # Store cost history
    J_history = []

    # Store parameter history
    p_history = []

    # Initialize w and b
    w = w_in
    b = b_in

    # Repeat gradient descent
    for i in range(num_iters):

        # Calculate gradients
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update b
        b = b - alpha * dj_db

        # Update w
        w = w - alpha * dj_dw

        # Save cost
        if i < 100000:
            J_history.append(
                cost_function(x, y, w, b)
            )

            # Save w and b
            p_history.append([w, b])

        # Print progress
        if i % math.ceil(num_iters / 10) == 0:

            print(
                f"Iteration {i:4}: "
                f"Cost {J_history[-1]:0.2e} "
                f"dj_dw: {dj_dw:0.3e}, "
                f"dj_db: {dj_db:0.3e} "
                f"w: {w:0.3e}, "
                f"b: {b:0.5e}"
            )

    return w, b, J_history, p_history


# ============================================================
# 5. Initialize Parameters
# ============================================================

w_init = 0
b_init = 0


# ============================================================
# 6. Gradient Descent Settings
# ============================================================

iterations = 10000

# Learning rate
tmp_alpha = 1.0e-2


# ============================================================
# 7. Run Gradient Descent
# ============================================================

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient
)


# ============================================================
# 8. Final Values
# ============================================================

print(
    f"(w,b) found by gradient descent: "
    f"({w_final:8.4f},{b_final:8.4f})"
)


# ============================================================
# 9. Predictions
# ============================================================

print(
    f"1000 sqft house prediction "
    f"{w_final * 1.0 + b_final:0.1f} Thousand dollars"
)

print(
    f"1200 sqft house prediction "
    f"{w_final * 1.2 + b_final:0.1f} Thousand dollars"
)

print(
    f"2000 sqft house prediction "
    f"{w_final * 2.0 + b_final:0.1f} Thousand dollars"
)