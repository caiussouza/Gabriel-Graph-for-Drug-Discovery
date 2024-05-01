import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class RBF:
    def __init__(self):
        """
        Instantiates a Radial Basis Function (RBF) network.

        For more information on Radial Basis Function Networks (RBFNs), reach [Radial Basis Function Networks(RBFNs)](https://mohamedbakrey094.medium.com/radial-basis-function-networks-rbfns-be2ec324d8fb)
        """
        pass

    def fit_model(self, X_train, y_train, centers, sigma, classification=False):
        """Fits the model based on features (X) and labels (y) values.

        ### Args:
            `X_train (matrix like)`: Training features.
            `y_train (array like)`: Training labels.
            `centers (array like)`: Centers used for calculating radial basis functions.
            `sigma (float)`: Sigma parameter.
            `classification (bool, optional)`: If True, fit the model based on a classification binary problem. Defaults to False.
        """
        if type(X_train) != np.ndarray:
            X_train = np.array(X_train)
        if classification:
            # Making sure that the labels are in a {+1, -1} range
            y_train = 2 * (y_train == 1) - 1
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
        """Performs a prediction based on learned parameters.

        ### Args:
            `X_test (matrix like)`: Testing features.
            `classification (bool, optional)`: If True, performs a classification prediction. Defaults to False.

        ### Returns:
            np.ndarray: array containing predictions.
        """
        if type(X_test) != np.ndarray:
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


def gaussian_rbf(x, center, sigma):
    """Gaussian kernel.

    Refer to: [Gaussian kernel](https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a)

    ### Args:
        `x (int, float or array-like)`: argument
        `center (int, float or array-like)`: center
        `sigma (int or float)`: sigma

    ### Returns:
        float or array-like: Result
    """
    return np.exp(-cdist([x], [center]) / (2 * sigma**2))
