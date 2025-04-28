#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>




int main(int argc, char **argv)
{
    int *v = malloc(sizeof(int));
    *v = 100;
    //agora está na heap 

    pid_t pid=fork(); // Funcao usada para criar um novo processo
    if(pid==0){// Processo filho
      *v+=100;
      printf("v = %d\n",*v);
      exit(0); // Funcao que encerra o processo
    }else if(pid>0){ // Processo pai
      wait(NULL); // O pai vai esperar a conclusao do filho
      printf("v = %d\n",*v);
    }
    printf("v = %d\n",*v);

    free(v);
    return 0;
}