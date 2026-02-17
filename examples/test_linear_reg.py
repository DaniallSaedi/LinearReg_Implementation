import numpy as np
import pandas as pd
from ..linear_reg import LinearReg

# Just for splitting data and use metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def make_linear_dataset(m=200, noise_std=0.5, seed=42):
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(-5, 5, size=m)
    x2 = 0.5 * x1 + rng.normal(0, 1, size=m)
    x3 = rng.normal(0, 3, size=m)

    X = np.column_stack([x1, x2, x3])

    theta_true = np.array([[2.0],  
                            [1.5],
                            [-3.0],
                            [0.7]])

    X_bias = np.hstack([np.ones((m, 1)), X])
    y = X_bias @ theta_true + rng.normal(0, noise_std, size=(m, 1))

    return X, y.ravel(), theta_true


X, y, true_theta = make_linear_dataset(m = 1000)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.25, random_state=42)

model = LinearReg()
model.fit(X_train,
          y_train,
          alpha = 0.01)



model2 = LinearReg()
model2.fit(X_train, 
           y_train,
           alpha = 0.01,
           scale=True)


print("Model Theta (without within scaling) : ")
print(model.theta)

print("Model Theta (with within scaling) : ")
print(model2.theta)

print("Real theta values : ")
print(true_theta)

print('r2 score : (No scaling)', end='')
print(r2_score(model.predict(X_test), y_test))

print('r2 score : (with scaling)', end='')
print(r2_score(model2.predict(X_test), y_test))
