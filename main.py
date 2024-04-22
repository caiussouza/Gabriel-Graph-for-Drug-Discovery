import Gabriel_Graph as GG
from sklearn.datasets import make_moons, make_circles

X, y = make_circles(250, noise=0.3, random_state=42)

grafo1 = GG.Gabriel_Graph(X, y, dist_method="euclidean")
grafo1.build_graph(wilson_editing=True, k=3)
grafo1.plot(label=True)
