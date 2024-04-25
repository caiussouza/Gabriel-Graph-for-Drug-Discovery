import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statistics import mode


def sign(x):
    """Sign function

    Args:
        x (int or float): Argument

    Returns:
        int: 1 if x is positive or zero, -1 if x is negative.
    """
    return 2 * (x >= 0) - 1


def two_classes_scatter(X, y, col_1="blue", col_2="red"):
    """Plots a scatter for binary bidimensional data

    Args:
        X (array like): input data
        y (array like): labels
        col_1 (str, optional): Class 1 color. Defaults to "blue".
        col_2 (str, optional): Class 2 color. Defaults to "red".
    """
    colors = y == 1
    colors = [col_1 if color == False else col_2 for color in colors]
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.show()


def make_gaussian(N1, M1, S1, N2, M2, S2, seed=42):
    """Generates two gaussian distributions on plane.

    Args:
        N1 (int): Samples on class 1.
        M1 (int or array like): Center of class 1.
        S1 (int or float): Standard deviation of class 1.
        N2 (int): Samples on class 2.
        M2 (int or array like): Center of class 2.
        S2 (int or float): Standard deviation of class 2.
        seed (int or float, optional): Seed for reproductibility. Defaults to 42.

    Returns:
        ndarray: Data (X) and labels (y).
    """
    Xc1 = np.random.normal(loc=M1, scale=S1, size=(N1, 2))
    Xc2 = np.random.normal(loc=M2, scale=S2, size=(N2, 2))
    y_c1 = np.full(N1, 0).reshape(-1, 1)
    y_c2 = np.full(N2, 1).reshape(-1, 1)
    Xy_c1 = np.hstack((Xc1, y_c1))
    Xy_c2 = np.hstack((Xc2, y_c2))
    Xy = np.vstack((Xy_c1, Xy_c2))
    np.random.shuffle(Xy)

    X = Xy[:, :-1]
    y = Xy[:, -1]
    y = y.astype(int)
    return X, y


def gaussian_rbf(x, center, sigma):
    return np.exp(-cdist([x], [center]) / (2 * sigma**2))


# Wilson editing externo para uso em datasets sem precisar construir grafos
def wilson_editing(X, y, k=1, distance="euclidean", plot=False):
    X = pd.DataFrame(X)
    y = 2 * (y == 1) - 1
    D = cdist(X, X, metric=distance)
    dist_vet = []
    Vp = []
    outliers = []
    for i in range(len(D)):
        dist_vet = D[i,]
        idx_knn = np.argsort(dist_vet)[: k + 1]
        idx_knn = np.delete(idx_knn, 0)
        k_nearest_classes = y[idx_knn]
        i_pred = sign(sum(k_nearest_classes))
        moda = mode(k_nearest_classes)
        i_pred = moda
        if y[i] == i_pred:
            Vp.append(np.hstack((X.iloc[i, :].values, y[i])))
        else:
            outliers.append(np.hstack((X.iloc[i, :].values, y[i])))

    Vp = pd.DataFrame(Vp)
    Xp = Vp.iloc[:, :-1]
    yp = Vp.iloc[:, -1]
    y_p_01_range = 1 * (yp >= 0)
    yp = y_p_01_range
    Xp = np.array(Xp)

    if outliers:
        outliers = pd.DataFrame(outliers)
        X_otl = outliers.iloc[:, :-1]
        y_otl = outliers.iloc[:, -1]
        y_otl_01_range = 1 * (y_otl >= 0)
        y_otl = y_otl_01_range
        X_otl = np.array(X_otl)
        has_otl = True
    else:
        X_otl = None
        y_otl = None
        has_otl = False

    if plot:
        if has_otl:
            plt.scatter(
                Xp[:, 0], Xp[:, 1], c=yp, cmap="coolwarm", label="Amostras confiáveis"
            )
            plt.scatter(
                X_otl[:, 0],
                X_otl[:, 1],
                c=y_otl,
                cmap="coolwarm",
                marker="+",
                label="Outliers",
            )
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()
        else:
            plt.scatter(Xp[:, 0], Xp[:, 1], c=yp, cmap="coolwarm", label="Amostras")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.show()
    else:
        return Xp, yp, X_otl, y_otl, has_otl
