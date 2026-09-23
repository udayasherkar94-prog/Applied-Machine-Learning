# House Price Prediction using Multiple Linear Regression

## 📌 Project Overview

This project predicts house prices using **Multiple Linear Regression** and **Gradient Descent**.

The model uses multiple features such as:

* House size in square feet
* Number of bedrooms
* Number of floors
* Age of the house

Feature scaling is performed using **Z-score normalization** to improve the performance and convergence of gradient descent.

## 🧠 Concepts Used

* Multiple Linear Regression
* Cost Function
* Gradient Descent
* Learning Rate
* Feature Scaling
* Z-score Normalization
* NumPy


## 📊 Features

The model uses four input features:

| Feature  | Description        |
| -------- | ------------------ |
| Size     | House size in sqft |
| Bedrooms | Number of bedrooms |
| Floors   | Number of floors   |
| Age      | Age of the house   |

## ⚙️ Z-score Normalization

Z-score normalization is performed using:

```text
X_norm = (X - μ) / σ
```

where:

* `μ` = mean of the feature
* `σ` = standard deviation of the feature

After normalization, the features are centered around zero with a similar scale.

## 🚀 Gradient Descent

Gradient descent is used to minimize the cost function and find the optimal values of:

* `w` → weights
* `b` → bias

The model uses:

```text
w = w - α × gradient
b = b - α × gradient
```

A learning rate of `0.1` is used after feature normalization.

## 🏠 Example Prediction

The model predicts the price of a house with:

```text
Size      = 1200 sqft
Bedrooms  = 3
Floors    = 1
Age       = 40 years
```

The new input is first normalized using the training-data mean and standard deviation and then passed to the trained model.

## 🛠️ Technologies Used

* Python
* NumPy
* Machine Learning --> supervised --> linear regression with multiple features

## 📁 Project Structure

House-Price-Prediction/
│
├── README.md
├── house_price_prediction.py
└── requirements.txt

## 🎯 Learning Outcome

Through this project, I learned how multiple linear regression works with multiple input features and how gradient descent can be improved using feature scaling.

I also learned why z-score normalization is useful when features have significantly different ranges.
