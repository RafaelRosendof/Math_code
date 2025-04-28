#include <pthread.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdlib.h>

// Variavel global que define o numero de threads
int numberOfThreads=5;



// Funcao que as threads vao usar para dar um ola
void *printHello(void *tid){
  printf("Ola, eu sou a thread %d.\n", (int)(size_t) tid);
  pthread_exit(NULL);

  pthread_mutex_lock(&lock)
}

int main(int argc, char **argv)
{
  pthread_t threads[numberOfThreads];
  for (int i=0;i<numberOfThreads;i++){
    // Criando uma thread e passando para ela a funcao printHello
    int status=pthread_create(&threads[i],NULL,printHello,(void *)(size_t) i);

    // Se o status nao for 0, houve um erro na criacao da thread
    if(status!=0){
      printf("Erro na criacao da thread %d. Codigo de erro %d.",i, status);
      return 1;
    }

    // Faz a thread principal esperar pela thread filha. Similar a funcao wait.
    pthread_join(threads[i],NULL);
    //pthread_

  }
  return 0;
}