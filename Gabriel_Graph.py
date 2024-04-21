import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from scipy.spatial.distance import cdist


class Gabriel_Graph:
    def __init__(self, X, y, index=None, dist_method="euclidean", palette="bright"):
        """Gabriel Graph initializer. It builds a Gabriel Graph based on an input
           matrix (X). It may also contains labels values (y) if working on supervised
           learning. The distances between points are calculated based on changeable methods.

        Args:
            X (pd.DataFrame or np.ndarray): Input matrix (without labels!)
            y (pd.DataFrame or np.ndarray): Label vector.
            index (array): Optional input matrix index.
            dist_method (str, optional): Distance method for calculating the graph. Defaults to 'euclidean'.
            palette (str, optional): Color palette for nodes. Defaults to 'bright'.
        """
        assert isinstance(
            X, (pd.DataFrame, np.ndarray)
        ), "X parameter is not a pandas DataFrame."
        if type(X) == np.ndarray:
            X = pd.DataFrame(X)
        self.X = X

        assert isinstance(
            y, (pd.DataFrame, np.ndarray)
        ), "y parameter is not a pandas DataFrame or numpy array."
        if type(y) == pd.DataFrame:
            y = np.array(y)
        self.y = y

        if index:
            self.index = index
        else:
            self.index = X.index

        assert isinstance(dist_method, str), "dist_method must be a string."
        self.dist_method = dist_method

        assert isinstance(palette, str), "palette_default must be a string"
        self.palette_deft = sns.color_palette(palette)

    def build_gabriel_graph(self):

        # Distance matrix
        D = cdist(self.X, self.X, metric=self.dist_method)

        GG = nx.Graph()

        for i in range(len(self.X)):
            label = self.y[i]
            GG.add_node(
                i,
                pos=self.X.iloc[i, :],
                label=label,
                id=self.X.index[i],
                color=self.palette_deft[label],
            )

        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):
                for k in range(len(self.X)):
                    is_GG = True
                    if (i != j) and (j != k) and (i != k):
                        if D[i, j] ** 2 > (D[i, k] ** 2 + D[j, k] ** 2):
                            is_GG = False
                            break
                if is_GG:
                    GG.add_edge(i, j)

        self.GGraph = GG
        self.node_locations = nx.get_node_attributes(GG, "pos")
        self.node_colors = list(nx.get_node_attributes(GG, "color").values())
        self.node_ids = nx.get_node_attributes(GG, "id")
        return GG

    def plot_2d_gg(self):
        nx.draw(
            self.GGraph,
            self.node_locations,
            node_color=self.node_colors,
            labels=self.node_ids,
            with_labels=True,
        )
        plt.show()

    def adjacency_matrix(self, sparse=False):

        adj_mat = nx.adjacency_matrix(self.GGraph)

        if sparse:
            return adj_mat
        if not sparse:
            adj_mat = pd.DataFrame(adj_mat.toarray())
            return adj_mat
