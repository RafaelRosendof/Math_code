import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def func(x):
    return np.sin(x)/x

def animate_quadratic_integration():
    fig, ax = plt.subplots()
    
    x = np.linspace(-20, 20, 100)  
    y = func(x) #conjunto imagem  
    
    # Inicializa um gráfico de linha vazio
    line, = ax.plot(x, y, 'b-')

    ax.axhline(y = 0 , color = 'white')
    ax.set_facecolor('black')

    # Função para atualizar o gráfico em cada quadro
    def update(frame):
        if frame == 0:
            return

        # Intervalo atual para integração
        x_interval = np.linspace(-20, x[frame], 100)
        y_interval = func(x_interval)
        ax.fill_between(x_interval, y_interval, alpha=0.5, color='red')

    ani = FuncAnimation(fig, update, frames=np.arange(len(x)), repeat=False, interval=100)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Integral não elementar: $f(x) = sin(x)/x$ = Si(x) + C')
    
    plt.show()


def main():
    animate_quadratic_integration()

if __name__ == '__main__':
    main()
