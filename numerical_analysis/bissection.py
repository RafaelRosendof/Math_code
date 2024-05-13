import numpy as np


class bissection():

    def function(self , x):
        return x**3 + np.cos(x)

    def method(self , a , b, erro):
        reps = 0
        if self.function(a) * self.function(b) > 0:
            return "Intervalo não tem raiz real"    

        while np.abs(b-a)  > erro:
            c = ( a + b )/2
            if np.abs(self.function(c)) <= erro:
                print("achou a raiz!!! ")
                return c
            
            elif(self.function(c) * self.function(a) < 0):
                b = c
            
            else:
                a = c
            reps += 1
            print("iteração" , reps ,"Com um erro de: ", self.function(c))
        return c
        
def main():
    print("Calculando a raiz do polinômio X*X*X + cos(X)")

    a = np.float128(input("Digite o primeiro valor do intervalo: "))
    b = np.float128(input("Digite o segundo valor do intervalo: "))
    erro = np.float128(input("Digite o erro: "))

    bis = bissection()
    bis.method(a,b,erro)



if __name__ =="__main__":
    main()