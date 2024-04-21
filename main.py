import Gabriel_Graph as GG
from sklearn.datasets import make_moons

X, y = make_moons(50, noise=0.1, random_state=42)

grafo1 = GG.Gabriel_Graph(X, y)
grafo1.build_gabriel_graph()
grafo1.plot_2d_gg()
