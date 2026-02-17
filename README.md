A lightweight and educational implementation of Linear Regression using Gradient Descent, written entirely with NumPy. This project is ideal for learning how linear regression works under the hood-without relying on scikit-learn.

Features:

    Gradient Descent optimization

    Optional feature scaling

    Optional L2 regularization (Ridge)

    Early stopping based on gradient norm

    Tracks loss history

    Clean and simple API (fit, predict)


Project Structure:

    LinearReg Implementation/
     |
     |__ linear reg.py
     |
     |__ examples/
         |__ test linear reg.py # Example usage

If you want to import it like a package:
    pip install -e .


How to run:
    python -m LinearReg_Implementation.examples.test_linear_reg
