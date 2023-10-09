import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da distribuição normal
mu = [0, 0, 0]  # Médias dos eixos x, y e z
sigma = [1, 1, 1]  # Desvios padrão dos eixos x, y e z

# Cria um conjunto de pontos (x, y, z) seguindo a distribuição normal multivariada
num_points = 1000
np.random.seed(0)
points = np.random.multivariate_normal(mu, np.diag(sigma), num_points)

# Cria um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plota os pontos no gráfico 3D
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Distribuição Normal')

# Defina rótulos para os eixos
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')

# Defina limites dos eixos, se necessário
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

# Exibe o gráfico
plt.legend()
plt.show()

