import numpy as np
import matplotlib.pyplot as plt

v1,v2,v3 = map(int,input("type the first vector: ").split())
z1,z2,z3 = map(int,input("type the second vector: ").split())

vector1 = np.array([v1,v2,v3])
vector2 = np.array([z1,z2,z3])
vector3 = vector2 * vector1

figura = plt.figure()

X = figura.add_subplot(111,projection='3d')

X.quiver(0,0,0 , vector1[0],vector1[1],vector1[2],color='r',label='Vector 1')
X.quiver(0,0,0 , vector2[0],vector2[1],vector2[2],color='g',label='Vector 2')
X.quiver(0,0,0, vector3[0],vector3[1],vector3[2],color='b',label='Vector 3')


X.set_xlim([-20,20])
X.set_ylim([-20,20])
X.set_zlim([-20,20])

X.grid(True)

plt.show()
