import core.Gabriel_Graph as GG
from core.utils import (
    two_classes_scatter,
    wilson_editing,
    plot_decision_surface,
    GGRBF_K_Fold_Performance,
    GGRBF_LOOCV_Accuracy,
    make_xor,
    make_gaussian,
)
from core.RBF import RBF
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.datasets import make_circles, make_moons, make_classification
import matplotlib.pyplot as plt

"""Teste do grafo de Gabriel e das RBF para problemas
   bidimensionais de classificação binária.
   A avaliação segue o seguinte roteiro:
   1) Escolha do dataset (xor, gaussianas etc.)
   2) Visualização dos dados pré-processamento
   3) Visualização dos dados após a edição de Wilson utilizando o k escolhido
   4) Construção e visualização do grafo utilizando dados de treinamento
   5) Obtenção dos SSVs e criação de RBF centrada nesses.
   6) Avaliação do modelo por meio da acurácia, AUC e MMC, bem como pela visualização do separador
   7) Avaliação por meio do KFold
"""

# Gerando os dados

# Duas gaussianas
X, y = make_gaussian(100, (2, 2), 2, 100, (4, 4), 2)

# Problema qualquer de classificação binário
# X, y = make_classification(
#     n_samples=200, n_features=2, n_classes=2, n_clusters_per_class=2, n_redundant=0
# )

# Problema das duas luas
# X, y = make_moons(200, noise=0.3, random_state=42)

# Problema do xor
# X, y = make_xor(50, 50, M11=(2, 2), M21=(4, 4), M12=(2, 4), M22=(4, 2), S1=0.7, S2=0.7)

# Dois círculos
# X, y = make_circles(200, noise=0.1, random_state=42)

# Conversão para dataframe para que o tipo se adeque ao esperado pelo grafo
X = pd.DataFrame(X)

# Plot dos dados "crus"
two_classes_scatter(X, y)

# Plot com a representação dos outliers de acordo com a Wilson editing
wilson_editing(X, y, k=2, plot=True)

# Divisão entre dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criação do grafo de Gabriel com dados de treinamento
grafo = GG.Gabriel_Graph(X_train, y_train)

# Para desativar a edição de Wilson, desabilite o parâmetro wilson_editing
grafo.build(wilson_editing=True, k=2)

# Alocação dos centros do grafo
rbf_centers = grafo.calculate_centers()

# Plot do grafo (se os dados forem bidimensionais)
grafo.plot(label=True, show_centers=True)

# Como a saída da RBF para classificação está no range {+1, -1}, os rótulos de teste também devem estar
y_test = 2 * (y_test == 1) - 1

# Construção e ajuste do modelo
model = RBF()
model.fit_model(X_train, y_train, rbf_centers, 1, classification=True)

# Predição em dados não vistos
y_hat = model.predict(X_test, classification=True)

# Métricas de desempenho do modelo
acc = accuracy_score(y_test, y_hat)
auc = roc_auc_score(y_test, y_hat)

print(f"A acurácia do modelo é {acc*100:.2f}%")
print(f"A AUC do modelo é {auc:.2f}")
print(f"O MMC do modelo é {matthews_corrcoef(y_test, y_hat):.2f}")

# Visualização da superfície de decisão
plot_decision_surface(X, y, model)

# Conversão do range para {+1, -1} para se adequar às saídas da RBF
y = 2 * (y == 1) - 1

# Avaliação por meio de KFold
mean, sd = GGRBF_K_Fold_Performance(
    X, y, K_kfold=10, wilson_editing=True, K_wilson=3, perf_metric="accuracy"
)
print(f"Acurácia K-Fold: {mean*100:.2f} +- {sd*100:.2f}")

# Avaliação por LOOCV (atualmente é inviável para datasets maiores)
# loocv_auc = GGRBF_LOOCV_Accuracy(X, y, wilson_editing=True, K_wilson=10)
# print(f"LOOCV: {loocv_auc*100:.2f}")
