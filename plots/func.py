import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#função z²+x²+y² = 9
def func():

    figure = plt.figure()

    #criando uma grade
    x = np.linspace(-6,6,1000)
    y = np.linspace(-6,6,1000)
    fx,fy = np.meshgrid(x,y)

    z = np.sqrt(9-fx**2 - fy**2)

    graf = figure.add_subplot(111,projection='3d')

    graf.plot_surface(fx,fy,z, cmap = 'viridis')
    print("Função 9 = z² + x² + y²")

    plt.show()


def parab():
    figas = plt.figure()

    fx = np.linspace(-3,3,1000)
    fy = np.linspace(-3,3,1000)
    x,y = np.meshgrid(fx,fy)

    z = np.sqrt(4 - x**2 - 4*y**2)

    grafico = figas.add_subplot(111,projection='3d')

    grafico.plot_surface(x,y,z ,cmap = 'Spectral')
    print("Função no r3 f(x,y)² = 4 - x² - 4y²")
    #grafico.set_box_aspect([1, 1, 0.5])
    plt.show()



def main():
    escolha = input("Escolha entre o primeiro e o segundo paraboloide: ")
    if escolha == '1':
        func()
    else:
        parab()

if __name__ == "__main__":
    main()

