from core.utils import sign
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from statistics import mode


class Gabriel_Graph:
    def __init__(self, X, y, index=None, dist_method="euclidean", palette="bright"):
        """## Initializer for Gabriel Graph.

        ### Args:
        - `X (pd.DataFrame)`: Input matrix containing features (without labels!).
        - `y (pd.DataFrame or np.ndarray)`: Label vector.
        - `index (array, optional)`: Index array for the input matrix. If not explicitly provided, DataFrames index will be used.
        - `dist_method (str, optional)`: Distance metric for calculating distances between samples (nodes) for generating the graph. Defaults to 'euclidean'.
        - `palette (str, optional)`: Color palette for node representation. Defaults to 'bright'.

        ### Attributes:
        - `X_`: DataFrame containing feature values.
        - `y_`: Numpy array containing label values.
        - `index_`: Array containing index values.
        - `dist_method_`: Metric used for calculating the graph. Read more on [Distance metrics in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
        - `palette_deft_`: Default palette used for plotting the graph. Read more on [Color palettes in Seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html)
        - `centers_`: Structural Support Vectors (SSV) or "Vetores de Suporte Estrutuais (VSE)". Read more on [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)
        - `GGraph_`: The Gabriel Graph itself represented by a networkx graph object.
        - `node_locations_`: Feature vector describing each sample (node) in the graph.
        - `node_colors_`: Color of each node. It is used for distinguishing classes or SSVs in cases when plotting is necessary.
        - `node_ids_`: Name of each node.
        - `Vp_`: V' (V prime) calculated based on Wilson editing algorithm. This represents the noise-cleaned DataFrame, containing both features and labels.

        ### Examples:

        Creating a Gabriel Graph using Wilson editing for noise reduction.

        ```
        # Importing the package
        >> import Gabriel_Graph as GG
        # Loading a dataset
        >> X, y = make_moons(200, noise=0.1, random_state=42)
        # Instantiating the graph
        >> graph = GG.Gabriel_Graph(X, y)
        # Building the graph and applying Wilson editing with a k=10
        >> graph.build(wilson_editing=True, k=10)
        # Calculating the centers (SSVs)
        >> graph.calculate_centers()
        # Plotting the graph (only is data is binary, bidimensional and labels are in a {1, 0} range)
        >> graph.plot(label=True, show_centers=True)
        ```
        """
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)
        self.X_ = X

        if type(y) != np.ndarray:
            y = np.array(y)
        self.y_ = y

        if index is not None:
            self.index_ = index
        else:
            self.index_ = X.index

        assert isinstance(dist_method, str), "dist_method must be a string."
        self.dist_method_ = dist_method

        assert isinstance(palette, str), "palette_default must be a string"
        self.palette_deft_ = sns.color_palette(palette)

        self.centers_ = None
        self.GGraph_ = None
        self.node_locations_ = None
        self.node_colors_ = None
        self.node_ids_ = None
        self.Vp_ = None

    def build(self, wilson_editing=False, k=1):
        """Constructs a Gabriel Graph, a proximity graph representing
        the relationships between points in the input matrix (X). If labels (y) are
        provided, it can be used for supervised learning tasks. Distances between
        points are calculated based on the method specified in the initializer.

        ### Args:
            - `wilson_editing (bool, optional)`: If True, implements Wilson editing algorithm for noise reduction. Defaults to False. Read more on [Using Representative-Based Clustering for Nearest Neighbor Dataset Editing](https://www.researchgate.net/profile/Ricardo-Vilalta/publication/4133603_Using_Representative-Based_Clustering_for_Nearest_Neighbor_Dataset_Editing/links/0f31753c55a8d611fc000000/Using-Representative-Based-Clustering-for-Nearest-Neighbor-Dataset-Editing.pdf?origin=publication_detail&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb25Eb3dubG9hZCIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19)

            - `k (int, optional)`: The k parameter for Wilson editing. Defaults to 1 (1-NN).
        """

        if wilson_editing:
            D = cdist(self.X_, self.X_, metric=self.dist_method_)
            dist_vet = []
            Vp = []

            for i in range(len(D)):
                dist_vet = D[i,]
                idx_knn = np.argsort(dist_vet)[: k + 1]
                idx_knn = np.delete(idx_knn, 0)
                k_nearest_classes = self.y_[idx_knn]
                moda = mode(k_nearest_classes)
                i_pred = moda
                if self.y_[i] == i_pred:
                    Vp.append(np.hstack((self.X_.iloc[i,].values, self.y_[i])))

            Vp = pd.DataFrame(Vp)
            self.Vp_ = Vp
            self.X_ = Vp.iloc[:, :-1]
            self.y_ = Vp.iloc[:, -1].astype(int)

        D = cdist(self.X_, self.X_, metric=self.dist_method_)

        GG = nx.Graph()

        for i in range(len(self.X_)):
            label = self.y_[i]
            GG.add_node(
                i,
                pos=self.X_.iloc[i, :],
                label=label,
                id=self.X_.index[i],
                color=self.palette_deft_[label],
            )

        for i in range(len(self.X_)):
            for j in range(i + 1, len(self.X_)):
                for k in range(len(self.X_)):
                    is_GG = True
                    if (i != j) and (j != k) and (i != k):
                        if D[i, j] ** 2 > (D[i, k] ** 2 + D[j, k] ** 2):
                            is_GG = False
                            break
                if is_GG:
                    GG.add_edge(i, j)

        self.GGraph_ = GG
        self.node_locations_ = nx.get_node_attributes(GG, "pos")
        self.node_colors_ = list(nx.get_node_attributes(GG, "color").values())
        self.node_ids_ = nx.get_node_attributes(GG, "id")

    def plot(self, label=True, show_centers=False):
        """Plots a 2D graph if data is binary, bidimensional and labels are in an {1, 0} range.

        ### Args:
            - `label (bool, optional)`: Presence of labels. Defaults to True.
        """
        if show_centers:
            assert self.centers_ is not None, "Centers were not calculated yet."
            color_aux_list = self.node_colors_.copy()
            for i in self.GGraph_.nodes:
                if i in self.centers_.index:
                    # color_aux_list[i] = 'gray' se quiser cinza
                    rgb_val = list(color_aux_list[i])
                    rgb_val[1] += 0.5
                    color_aux_list[i] = rgb_val

        nx.draw(
            self.GGraph_,
            self.node_locations_,
            node_color=color_aux_list if show_centers else self.node_colors_,
            labels=self.node_ids_,
            with_labels=label,
            alpha=0.8,
            edgecolors="black",
        )
        plt.show()

    def adjacency_matrix(self, sparse=False):
        """Adjacency matrix representation of the graph. Is useful for
        for visualizing graphs with dimensions greater than 2 or 3.

        ### Args:
            - `sparse (bool, optional)`: If True returns a sparse matrix in scipy. If False, returns a pandas DataFrame. Defaults to False.

        ### Returns:
            - `pandas DataFrame or scipy sparse matrix`: Adjacency matrix.
        """
        adj_mat = nx.adjacency_matrix(self.GGraph_)

        if sparse:
            return adj_mat
        if not sparse:
            adj_mat = pd.DataFrame(adj_mat.toarray())
            return adj_mat

    def calculate_centers(self):
        """Calculates the Structural Support Vectors (SSV) or "Vetores de Suporte Estrutuais (VSE)". Read more on [Dissertação Matheus Salgado](https://repositorio.ufmg.br/bitstream/1843/RAOA-BCFHQJ/1/matheus_salgado_dissertacao__1_.pdf)

        ### Returns:
            - `pd.DataFrame`: Vector of centers (SSVs)
        """
        edges = list(self.GGraph_.edges())

        node_pos = []
        node_labels = []
        for i in self.GGraph_.nodes:
            node_pos.append(self.GGraph_.nodes[i]["pos"])
            node_labels.append(self.GGraph_.nodes[i]["label"])
        node_pos = pd.DataFrame(node_pos)
        node_labels = pd.DataFrame(node_labels)

        centers = []
        for i in range(len(edges)):
            x1 = edges[i][0]
            x2 = edges[i][1]
            if self.GGraph_.nodes[x1]["label"] != self.GGraph_.nodes[x2]["label"]:
                centers.append(node_pos.iloc[x1, :])
                centers.append(node_pos.iloc[x2, :])
        centers = pd.DataFrame(centers)
        centers = centers.drop_duplicates()
        self.centers_ = centers
        return centers
