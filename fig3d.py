import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def cone():
    # Cria uma grade de pontos no espaço tridimensional
    phi = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, 5, 100)
    PHI, Z = np.meshgrid(phi, z)

# Calcula as coordenadas x, y e z de um cone
    X = Z * np.cos(PHI)
    Y = Z * np.sin(PHI)
                                #melhorar o código para entrar uma função 
# Cria um gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# Plota a superfície do cone
    ax.plot_surface(X, Y, Z, cmap='viridis')

# Exibe o gráfico
    plt.show()

def paraboloide():
    # Cria uma grade de pontos no espaço tridimensional
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

# Calcula a altura do paraboloide em cada ponto da grade
    Z = X**2 + Y**2

# Cria um gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# Plota a superfície do paraboloide
    ax.plot_surface(X, Y, Z, cmap='viridis')

# Exibe o gráfico
    plt.show()

def main():
    print("Escolha qual o gráfico 1 para cone 2 para paraboloide: ")
    escolha = input()
    
    if escolha == '1':
        cone()
    else:
        paraboloide()

if __name__ == "__main__":

    main()