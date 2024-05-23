import numpy as np
import math 

def GaussSiedel(a , b, chute , erro , maxIter):
    n = len(a)
    x = chute.copy()
    novoX = np.zeros(n)
    
    for iter in range(maxIter):
        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += a[i][j] * x[j]
            novoX[i] = (b[i] - soma) /a[i][i] #atualizando 

        if np.linalg.norm(novoX - x , np.inf) < erro:
            return novoX
        print('\n iteração' , iter , ' com o erro de: ' , novoX - x)   
        x = novoX.copy()
    
    raise ValueError("Deu merda negão") 
    
def main():
    print("Fatoração Gauss Siedel ")

    a = np.array([[10 , 1 , -1] ,
                  [2, 10, 8],
                 [7 , 1, 10]])
    
    b = np.array([ 10 , 20 , 30])
    
    chute = np.array([234,456,456])
    
    erro = np.float128(0.0000000056)
    
    max_iter = 1000

    try:
        sol = GaussSiedel(a , b , chute , erro , max_iter)
        print("Deu bom aqui ", sol)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()