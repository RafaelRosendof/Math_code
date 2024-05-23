import numpy as np 
from scipy.linalg import cholesky

matA = np.matrix('1 1 0 ; 1 2 1 ; 0 -1 3')

matB = np.matrix([[np.sum(matA[0,:])] , [np.sum(matA[1,:])] , [np.sum(matA[2,:])]])

print(matB)

U = cholesky(matA)
print(U)

ut = U.transpose()

Atest = ut.dot(U)

print(' Atest = \n',Atest)

print('\n\n')

d = np.linalg.solve(ut,matB)
x = np.linalg.solve(U,d)

print('\n\n Solution é \n',x)
