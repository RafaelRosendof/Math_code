import os , time , multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def F(x):
    return np.log10(np.sqrt(x*x+x) + 1)

#para calcular a integral em segumentos 
def trapz_seg(a , b ,n):
    #Calculo do H
    h = (b-a)/n
    #Lembrando que a regra do trapézio é assim
    integral = 0.5 * (F(a) + F(b))
    
    #Por fim estamos fatiando a integral em n segmentos de 1 até N e incrementando 
    for i in range(1 , n):
        integral += F(a + i*h)

    return integral * h

def processoParallel(a , b, n , threads):
    futures = []
    segmentos = (b - a) / threads

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(threads):
            seg_a = a + i * segmentos
            seg_b = a + (i + 1) * segmentos

            #n // numero de threads para cada segmento

            futures.append(executor.submit(trapz_seg , seg_a , seg_b , n // threads))

    #somando os resultados das threads
    total = sum(future.result() for future in futures)
    return total

def temporizador(a , b , n , threads):
    '''
    iniciar o cronômetro 
    calcula a função 
    finaliza o cronômetro
    tempo total
    '''
    start = time.time()
    res = processoParallel(a , b , n , threads)
    end = time.time()

    total = end - start
    return res , total

def main():
    
    print("Calculando a integral de log10(sqrt(x*x+x) + 1) de 0 a 1000 \n")

    num_cores = multiprocessing.cpu_count()
    print(f"Numero de cores: {num_cores} \n\n")

    a = 0 
    b = 1000
    n = 10000000

    res , total = temporizador(a , b , n , num_cores)

    print(f"Resultado: {res}")
    print(f"Tempo total: {total}")
   

if __name__ == "__main__":
    main()