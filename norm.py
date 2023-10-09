import numpy as np
import matplotlib.pyplot as plt

v1,v2,v3 = map(int,input("type the first vector: ").split())

Vector = np.array([v1,v2,v3])

vNorm = np.linalg.norm(Vector)

v_unit = Vector/vNorm

figura = plt.figure()

X = figura.add_subplot(111,projection='3d')

X.quiver(0,0,0, Vector[0],Vector[1],Vector[2],color='g',label='Vector')
X.quiver(0,0,0,v_unit[0],v_unit[1],v_unit[2],color='r',label='Norma')

X.set_xlim([-20,20])
X.set_ylim([-20,20])
X.set_zlim([-20,20])

X.grid(True)
X.text(Vector[0], Vector[1], Vector[2], f'Norm: {vNorm:.2f}', color='b')
X.legend()
plt.show()