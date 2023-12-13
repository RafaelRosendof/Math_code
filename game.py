#fazer um jogo de advinhação em python
import random

def gerador():
    chave = random.randint(1,101)
    return chave


def main():
    print("Parado aí vc precisa jogar um jogo, escolha a sua dificuldade: facil(0), médio(1), Dificil(2) ")
    dificuldade = input()
    if dificuldade == '0' :
        tentativas = 10
    elif dificuldade == '1':
        tentativas = 8
    else:
        tentativas = 4

    print("O senhor tem:",tentativas)

    numero_magico = gerador()


    print("O numero mágico está entre 1 e 100: Advinhe se possível: Voçê tem ",tentativas )

    while(tentativas > 0):
        print("O numero mágico está entre 1 e 100: Advinhe se possível: Voçê tem ",tentativas )

        palpite = int(input("Seu número: "))

        if palpite == numero_magico:
            print("Voce venceu")
            break
        elif palpite > numero_magico:
            print("Número maior")
            continue
        else:
            print("númeor menor")


        tentativas = tentativas -1

    if tentativas == 0:
        print("tenta da proxima, suas vidas acabaram ")

if __name__ == "__main__":
    main()