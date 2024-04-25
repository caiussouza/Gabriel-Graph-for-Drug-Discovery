import core.Gabriel_Graph as GG
from core.utils import make_gaussian, two_classes_scatter, wilson_editing
from core.RBF import RBF
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_circles, make_moons

import matplotlib.pyplot as plt

"""Teste do grafo de Gabriel e das RBF para problemas
   bidimensionais de classificação binária.
"""

# Gerando os dados
# X, y = make_gaussian(150, (2, 2), 1.5, 100, (4, 4), 1.5)
# X, y = make_moons(200, noise=0.4, random_state=42)
X, y = make_circles(200, noise=0.1, random_state=42)

# Plot dos dados "crus"
two_classes_scatter(X, y)
# Plot com a representação dos outliers de acordo com a Wilson editing
wilson_editing(X, y, k=10, plot=True)

# Divisão entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criação do grafo de Gabriel com dados de treinamento
grafo = GG.Gabriel_Graph(X_train, y_train)
# Para desativar a edição de Wilson, desabilite o parâmetro wilson_editing
grafo.build_graph(wilson_editing=True, k=10)
# Alocação dos centros do grafo
rbf_centers = grafo.get_centers()
# Plot do grafo (se os dados forem bidimensionais)
grafo.plot(label=True, show_centers=True)

# Para boa utilização da rede RBF, rótulos devem pertencer a {-1, 1}
y_train = 2 * (y_train == 1) - 1
y_test = 2 * (y_test == 1) - 1

# Construção e ajuste do modelo
model = RBF()
model.fit_model(X_train, y_train, rbf_centers, 1)
# Predição em dados não vistos
y_hat = model.predict(X_test, classification=True)
# Métricas de desempenho do modelo
acc = accuracy_score(y_test, y_hat)
auc = roc_auc_score(y_test, y_hat)

print(f"A acurácia do modelo é {acc*100}%")
print(f"A AUC do modelo é {auc}")

# Plot da superfície de separação
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()], classification=True)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.6)
colors = ["blue" if i == 0 else "red" for i in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
