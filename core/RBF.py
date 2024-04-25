import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.utils import gaussian_rbf


class RBF:
    def __init__(self):
        pass

    def fit_model(self, X_train, y_train, centers, sigma):
        if type(X_train) == pd.DataFrame:
            X_train = np.array(X_train)
        # Os rótulos precisam estar no range {-1,1}
        N = X_train.shape[0]
        n = X_train.shape[1]
        n_centers = centers.shape[0]
        mus = centers.values

        Phi = np.zeros((N, n_centers + 1))
        for lin in range(N):
            Phi[lin, 0] = 1
            for col in range(n_centers):
                Phi[lin, col + 1] = gaussian_rbf(X_train[lin, :], mus[col, :], sigma)
        w = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y_train

        self.w = w
        self.mus = mus
        self.sigma = sigma

    def predict(self, X_test, classification=False):
        if type(X_test) == pd.DataFrame:
            X_test = np.array(X_test)
        N = X_test.shape[0]
        pred = np.repeat(self.w[0], N)

        for j in range(N):
            for k in range(len(self.mus)):
                pred[j] += self.w[k + 1] * gaussian_rbf(
                    X_test[j, :], self.mus[k, :], self.sigma
                )
        if classification:
            pred = np.sign(pred)
        return pred
