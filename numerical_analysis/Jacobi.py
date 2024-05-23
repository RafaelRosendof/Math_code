import numpy as np
import math 

def jacobi(a , b , chute , erro , maxIter):
    n = len(a)
    x = chute.copy()
    #novoX = np.array(x)
    novoX = np.zeros(n)

    for iter in range(maxIter):
        for i in range(n):
            s = sum(a[i][j] * x[j] for j in range(n) if j != i)
            novoX[i] = (b[i] - s) / a[i][i] 
            

        if np.linalg.norm(novoX - x , ord=np.inf ) < erro:
            return novoX
        
        print('\n iteração' , iter , ' com o erro de: ' , novoX - x)
        x = novoX.copy()
        
    raise Exception("Cara não convergiu ")

        
def main():
    #DEFINING JACOBI SYSTEM
    a = np.array([[10 , 1 , -1] ,
                  [2, 10, 8],
                 [7 , 1, 10]])
    
    b = np.array([ 10 , 20 , 30])
    
    chute = np.array([234,456,456])
    
    erro = np.float128(0.0000000056)
    
    max_iter = 1000
    
    try:
        sol = jacobi(a , b , chute , erro , max_iter)
        print("Deu bom aqui ", sol)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
    