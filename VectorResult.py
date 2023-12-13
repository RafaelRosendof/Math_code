import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def vector():
    v1,v2,v3 = map(int,input("type the first vector: ").split())
    z1,z2,z3 = map(int,input("type the second vector: ").split())

    vector1 = np.array([v1,v2,v3])
    vector2 = np.array([z1,z2,z3])

    return vector1, vector2

def create_animation(vector1, vector2):
   
    vector3 = np.cross(vector1,vector2)
    def update(frame):
        X.cla()  # Limpa o gráfico a cada quadro

        # Vetor 1
        if frame >= 1:
            X.quiver(0, 0, 0, vector1[0], vector1[1], vector1[2], color='r', label='Vector 1')

        # Vetor 2
        if frame >= 2:
            X.quiver(0, 0, 0, vector2[0], vector2[1], vector2[2], color='g', label='Vector 2')

        # Vetor Resultante
        if frame >= 3:
            X.quiver(0, 0, 0, vector3[0], vector3[1], vector3[2], color='b', label='Vector 3')

        X.set_xlim([-20, 20])
        X.set_ylim([-20, 20])
        X.set_zlim([-20, 20])

        X.grid(True)

    figura = plt.figure()
    X = figura.add_subplot(111, projection='3d')

    ani = FuncAnimation(figura, update, frames=32, repeat=False, interval=200)
    plt.show()

def main():
    vector1 , vector2 = vector()
    create_animation(vector1,vector2)

if __name__ == '__main__':
    main()