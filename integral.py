import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função quadrática com concavidade para baixo: y = -x^2
def quadratic_function(x):
    return x**2

def animate_quadratic_integration():
    fig, ax = plt.subplots()
    
    x = np.linspace(-5, 5, 100)  # Intervalo de x
    y = quadratic_function(x)    # Valor da função quadrática
    
    # Inicializa um gráfico de linha vazio
    line, = ax.plot(x, y, 'b-')

    # Função para atualizar o gráfico em cada quadro
    def update(frame):
        if frame == 0:
            return

        # Intervalo atual para integração
        x_interval = np.linspace(-5, x[frame], 100)
        y_interval = quadratic_function(x_interval)
        ax.fill_between(x_interval, y_interval, alpha=0.5, color='blue')

    ani = FuncAnimation(fig, update, frames=np.arange(len(x)), repeat=False, interval=100)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Integração da função quadrática: $y = x^2$ (concavidade para baixo)')
    
    plt.show()


def main():
    animate_quadratic_integration()

if __name__ == '__main__':
    main()
