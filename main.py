import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# G = nx.from_numpy_array(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))


# G = nx.Graph()
# G.add_edge(1, 2)
# G.add_edge(2, 3)
# G.add_edge("A", "B")
# G.add_edge(2, "A")

EL = [(1, 2), (2, 1), (2, 3), (2, 2), ("A", "B"), (2, "A"), (print, "B")]
# G = nx.from_edgelist(EL)

G = nx.DiGraph()
G.add_edges_from(EL)
print(dict(G.degree)["A"])

# print(nx.adjacency_matrix(G))

nx.draw_planar(G, with_labels=True)
plt.show()
