import numpy as np
import sympy as sp


def LU(mat , n):

    lower = [[0 for x in range(n)]
             for y in range(n)]

    upper = [[0 for x in range(n)]
             for y in range(n)]
        
    for i in range(n):

        for k in range(i,n):
            soma = 0

            for j in range(i):
                soma += (lower[i][j] * upper[j][k])

            upper[i][k] = mat[i][k] - soma

        for k in range(i , n):
            if(i==k):
                lower[i][i] = 1
            else:

                soma = 0
                for j in range(i):
                    soma+= (lower[k][j] * upper[j][i])

                lower[k][i] = int((mat[k][i] - soma)/ upper[i][i])

    print("Triangular superior e inferior ")

    for i in range(n):
        for j in range(n):
            print(lower[i][j] , end="\t")
        print("", end="\t")

        for j in range(n):
            print(upper[i][j] , end="\t")
        print("")

mat = [[2, -1, -2],
       [-4, 6, 3],
       [-4, -2, 8]]
 
LU(mat, 3)