import numpy as np

class Newton():

    def func(self , x):
        return x**2 - np.sin(x)
    
    def deriv_func(self,x):
        return 2*x - np.cos(x)
    
    def newton(self , x , erro):

        #h = self.func(x)/self.deriv_func(x)
        iter = 0

        while True:
            h = self.func(x)/self.deriv_func(x)

            print("Iteração número", iter , "Com a iteração em ", h)

            if np.abs(h) < erro:
                break

            x = x-h
            iter +=1
        
        print("Valor da raiz é ", self.func(h))


def main():
    print("Calculando a raiz de X*X - sin(x) ")

    x = np.float128(input("Digite o X: "))
    erro = np.float128(input("Digite o erro: "))

    new = Newton()

    new.newton(x,erro)

if __name__ == "__main__":
    main()
