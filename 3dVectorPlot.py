import numpy as np
import matplotlib.pyplot as plt

#coloque as cordenadas do array

figas = np.array([4,2,4])

fig = plt.figure()
bigX = fig.add_subplot(111,projection='3d')

#define quais parâmetros que vão sair 

x_cord = [0,figas[0]]
y_cord = [0,figas[1]]
z_cord = [0,figas[2]]

bigX.plot(x_cord,y_cord,z_cord)

#define a abrangência do sub-espaço

bigX.set_xlim([-6,6])
bigX.set_ylim([-6,6])
bigX.set_zlim([-6,6])

bigX.grid(True)

plt.show()