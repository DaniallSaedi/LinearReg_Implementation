import numpy as np
import pandas as pd

class LinearReg:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.loss_history = []
        

    def fit(self, 
            X, 
            y, 
            alpha = 0.001, 
            lam = 0.0, 
            max_iter = 1000,
            scale = False,
            tol = 1e-8
            ):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y = y.reshape(-1, 1)

        m, n = X.shape
        
        # Scaling( if necessary)
        if scale:
            self.mu = X.mean(axis = 0)
            self.sigma = X.std(axis = 0) + 1e-8
            X = (X - self.mu) / self.sigma
        else:
            self.mu = np.zeros(n)
            self.sigma = np.ones(n)

        # Thetas initalization
        self.theta = np.zeros((n+1, 1))
        
        # Bias
        ones = np.ones((m, 1))
        X = np.hstack([ones, X])


        # Gradient Descent
        for _ in range(max_iter):

            error = X @ self.theta - y # m x 1
            
            grad = 1/m * X.T @ error # n+1 x 1
            grad[1:] += (lam/m) * self.theta[1:]

            self.theta -= alpha * grad


            loss = (1/(2*m)) * np.sum(error**2)
    
            self.loss_history.append(loss)


            if np.linalg.norm(grad) < tol:
                break

                
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = (X - self.mu) / self.sigma
        
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

        return X @ self.theta