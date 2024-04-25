import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from core.utils import sign


class Gabriel_Graph:
    def __init__(self, X, y, index=None, dist_method="euclidean", palette="bright"):
        """Gabriel Graph initializer.
        Args:
            X (pd.DataFrame or np.ndarray): Input matrix (without labels!)
            y (pd.DataFrame or np.ndarray): Label vector.
            index (array): Optional input matrix index.
            dist_method (str, optional): Distance method for calculating the graph. Defaults to 'euclidean'.
            palette (str, optional): Color palette for nodes. Defaults to 'bright'.
        """
        assert isinstance(
            X, (pd.DataFrame, np.ndarray)
        ), "X parameter is not a pandas DataFrame or numpy array."
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

        self.centers = None

    def build_graph(self, wilson_editing=False, k=1):
        """It builds a Gabriel Graph based on an input matrix (X).
        It may also contain labels values (y) if working on supervised
        learning. The distances between points are calculated based on input method.


        Args:
            wilson_editing (bool, optional): Implements a Wilson editing for noise reduction. Defaults to False. See more on: https://www.researchgate.net/profile/Ricardo-Vilalta/publication/4133603_Using_Representative-Based_Clustering_for_Nearest_Neighbor_Dataset_Editing/links/0f31753c55a8d611fc000000/Using-Representative-Based-Clustering-for-Nearest-Neighbor-Dataset-Editing.pdf?origin=publication_detail&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb25Eb3dubG9hZCIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19
            k (int, optional): k parameter for Wilson editing. Defaults to 1 (1-NN).

        Returns:
            nx.Graph: Gabriel Graph.
        """

        if wilson_editing:
            self.y = 2 * (self.y == 1) - 1
            D = cdist(self.X, self.X, metric=self.dist_method)
            dist_vet = []
            Vp = []

            for i in range(len(D)):
                dist_vet = D[i,]
                idx_knn = np.argsort(dist_vet)[: k + 1]
                idx_knn = np.delete(idx_knn, 0)
                k_nearest_classes = self.y[idx_knn]
                i_pred = sign(sum(k_nearest_classes))
                if self.y[i] == i_pred:
                    Vp.append(np.hstack((self.X.iloc[i,].values, self.y[i])))

            Vp = pd.DataFrame(Vp)
            self.X = Vp.iloc[:, :-1]
            self.y = Vp.iloc[:, -1]
            y_01_range = 1 * (self.y >= 0)
            self.y = y_01_range

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

    def plot(self, label=True, show_centers=False):
        """Plots a 2D graph if data is bidimensional.

        Args:
            label (bool, optional): Presence of labels. Defaults to True.
        """
        if show_centers:
            assert self.centers is not None, "Centers were not calculated yet."
            color_aux_list = self.node_colors.copy()
            for i in self.GGraph.nodes:
                if i in self.centers.index:
                    # color_aux_list[i] = 'gray' se quiser cinza
                    rgb_val = list(color_aux_list[i])
                    rgb_val[1] += 0.5
                    color_aux_list[i] = rgb_val

        nx.draw(
            self.GGraph,
            self.node_locations,
            node_color=color_aux_list if show_centers else self.node_colors,
            labels=self.node_ids,
            with_labels=label,
        )
        plt.show()

    def adjacency_matrix(self, sparse=False):
        """Adjacency matrix representation of the graph. Is useful for
        for visualizing graphs with dimensions greater than 2 or 3.

        Args:
            sparse (bool, optional): If True returns a sparse matrix in scipy.
            If False, returns a pandas DataFrame. Defaults to False.

        Returns:
            pandas DataFrame or scipy sparse matrix: Adjacency matrix.
        """
        adj_mat = nx.adjacency_matrix(self.GGraph)

        if sparse:
            return adj_mat
        if not sparse:
            adj_mat = pd.DataFrame(adj_mat.toarray())
            return adj_mat

    def get_centers(self):
        edges = list(self.GGraph.edges())

        node_pos = []
        node_labels = []
        for i in self.GGraph.nodes:
            node_pos.append(self.GGraph.nodes[i]["pos"])
            node_labels.append(self.GGraph.nodes[i]["label"])
        node_pos = pd.DataFrame(node_pos)
        node_labels = pd.DataFrame(node_labels)

        centers = []
        for i in range(len(edges)):
            x1 = edges[i][0]
            x2 = edges[i][1]
            if self.GGraph.nodes[x1]["label"] != self.GGraph.nodes[x2]["label"]:
                centers.append(node_pos.iloc[x1, :])
                centers.append(node_pos.iloc[x2, :])
        centers = pd.DataFrame(centers)
        centers = centers.drop_duplicates()
        self.centers = centers
        return centers
