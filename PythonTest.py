'''
O script em python abaixo será um método para realizar o calculo de uma integral numérica 
Dado uma função qualquer e um intervalor de integração, o método irá calcular a integral numérica
O diferencial será o uso de uma abordagem paralela para realizar o cálculo mais rapidamente
log10(sqrt(x*x+x) + 1)
'''
import numpy as np
import time , multiprocessing
from concurrent.futures import ThreadPoolExecutor



def F(x):
    return np.log10(np.sqrt(x*x + x) + 1)

def trapezio(a , b, n):
    h = (b-a)/n

    integral = 0.5 * (F(a) + F(b))

    #fatiar a integral em partes de 1 até N
    for i in range(1, n):
        integral += F(a + i*h)

    return integral * h

def trapzerio_parello(a , b , n , nthreads):
    futures = []
    segmentos = (b-a) / nthreads

    with ThreadPoolExecutor(max_workers = nthreads) as executor:
        for i in range(nthreads):
            seg_a = a + i * segmentos
            seg_b = a + (i+1) * segmentos

            #
            futures.append(executor.submit(trapezio, seg_a, seg_b, n//nthreads))

    total = sum(future.result() for future in futures)
    return total


def temporizador(a , b ,n , nthreads):
    '''
    Inicar o tempo 
    vamos calcular a integral
    end tempo 
    resultado
    '''
    start = time.time()
    resultado = trapzerio_parello(a , b , n , nthreads)
    end = time.time()
    total = end - start

    return resultado , total

def main():
    
    #INICIAR AS VARIAVEIS 
    a = 0 
    b = 1000
    n = 10000000

    cores = multiprocessing.cpu_count()

    print(f"Números de cores: {cores}")
    

    res , tempo = temporizador(a , b , n , cores )

    print(f"Resultado: {res}")
    print(f"Tempo: {tempo}")

if __name__ == "__main__":
    main()