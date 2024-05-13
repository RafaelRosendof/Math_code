import numpy as np

class secante:
    def __init__(self, x0, x1, erro):
       self.x0 = x0 
       self.x1 = x1
       self.erro = erro
        
    def fun(self, x):
        return x**3 - np.cos(x)

    def iterativo(self):
        iter = 0
        x_new = 0.0

        while True:
            x_new = self.x1 - (self.fun(self.x1) * (self.x1 - self.x0) / (self.fun(self.x1) - self.fun(self.x0)))

            if np.abs(self.fun(x_new)) < self.erro:
                break

            self.x0 = self.x1
            self.x1 = x_new
            print("Iteração: " , iter , "raiz aproximada: " , x_new , "    ")
            iter +=1
        return x_new

def main():
    a = float(input("Digite o lado esquerdo da busca: "))
    b = float(input("Digite o lado direito da busca: "))

    erro = np.float128(input("Digite o erro: "))

    secs = secante(a,b,erro)

    figas = secs.iterativo()

    print("FINALIZADO     ")

if __name__ == "__main__":
    main()
