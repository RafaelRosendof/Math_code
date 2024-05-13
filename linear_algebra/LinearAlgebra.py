import numpy as np
import matplotlib.pyplot as plt

class LA:
    def soma_vet(self):
        v1,v2,v3 = map(int,input("type the first vector: ").split())
        z1,z2,z3 = map(int,input("type the second vector: ").split())
    
        vector1 = np.array([v1,v2,v3])
        vector2 = np.array([z1,z2,z3])

        vectorRes = vector1 + vector2

        fig = plt.figure()
        X = fig.add_subplot(111, projection='3d')
        X.quiver(0,0,0 , vector1[0],vector1[1],vector1[2], color = 'b' , label='Vector 1')   
        X.quiver(0,0,0 , vector2[0],vector2[1],vector2[2], color = 'g' , label='Vector 2')
        X.quiver(0,0,0 , vectorRes[0],vectorRes[1],vectorRes[2], color = 'r' , label='Result')

        X.set_xlim([-20,20])
        X.set_ylim([-20,20])
        X.set_zlim([-20,20])

        X.grid(True)

        plt.show()

    def crossP(self):
        v1,v2,v3 = map(int,input("type the first vector: ").split())
        z1,z2,z3 = map(int,input("type the second vector: ").split())
    
        vector1 = np.array([v1,v2,v3])
        vector2 = np.array([z1,z2,z3])

        vectorRes = np.cross(vector1,vector2)

        fig = plt.figure()
        X = fig.add_subplot(111, projection='3d')
        X.quiver(0,0,0 , vector1[0],vector1[1],vector1[2], color = 'r' , label='Vector 1')   
        X.quiver(0,0,0 , vector2[0],vector2[1],vector2[2], color = 'b' , label='Vector 2')
        X.quiver(0,0,0 , vectorRes[0],vectorRes[1],vectorRes[2],color = 'g', label='Result')

        X.set_xlim([-20,20])
        X.set_ylim([-20,20])
        X.set_zlim([-20,20])

        X.grid(True)

        plt.show()

    def norma(self):
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

class options:

    def __init__(self):
        self.la = LA()

    def escolha(self):
        while True:
            print("Choose an option:")
            print("1. Vectorial Sum")
            print("2. Vectorial Dot")
            print("3. Vector Norm")
            print("4. Get out: ")
            choice = input("Enter your choice: ")
            if choice == "1":
                self.la.soma_vet()
            elif choice == "2":
                self.la.crossP()
            elif choice ==  "3":
                self.la.norma()
            elif choice == "4":
                break
            else:
                "Wrong choise "


def main():
    print("Seja bem vindo ao mundo da algebra linear: ")
    print()
    print("Be welcome to the Linear Algebra World: ")
    print("Agora voce pode esolher algumas opções matemáticas: ")

    figas = options()
    figas.escolha()


if __name__ == "__main__":
    main()
        
    