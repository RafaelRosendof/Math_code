import scipy as sc
import numpy as np

matA = np.matrix('8 -6 2 ; -4 11 7 ; 4 -7 6')

vecB = np.matrix('28 ; -40 ; 33')

P,L,U = sc.linalg.lu(matA)

print('L =\n' , L)
print('U =\n' , U)
print('P =\n' , P)

print(L.dot(U))

d = np.linalg.solve(L,vecB)
x = np.linalg.solve(U,d)

print("\n\n")
print(x)

