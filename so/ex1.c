#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdlib.h>

int main(int argc , char **argv){

    int segment_id = shmget(IPC_PRIVATE , sizeof(int) , S_IRUSR | S_IWUSR);
    if(segment_id == -1){
        perror("Erro ao criar segmento de memória compartilhada ");
        return 1;
    }

    //fazendo pointer compartilhado 
    int *shrPointer;

    shrPointer = (int *) shmat(segment_id , NULL , 0);
    if(shrPointer == (int *)-1){
        perror("Erro ao anexar memória compartilhada");
        return 1;
    }

    *shrPointer = 0;
    //fazendo filho 1

    pid_t pid1 = fork();

    if(pid1 < 0){
        perror("Erro ao inicar f1");
        return 1;
    } else if( pid1 == 0){
        shrPointer = (int *)shmat(segment_id , NULL , 0);
        *shrPointer += 10;
        printf("Primeiro filho (PID %d): incrementou o valor para %d\n", getpid(), *shrPointer);

        shmdt(shrPointer);

        exit(0);
    }
    else{

        pid_t pid2 = fork();

        if(pid2 < 0){
            perror("Erro ao inicar f1");
            return 1;
        }
        else if( pid2 == 0){
            shrPointer = (int *)shmat(segment_id , NULL , 0);
            *shrPointer += 10;
            printf("segundo filho (PID %d): incrementou o valor para %d\n", getpid(), *shrPointer);

            shmdt(shrPointer);

            exit(0);
        }
        else { 
            // Esperando 
            waitpid(pid1, NULL, 0);

            // Esperando 
            waitpid(pid2, NULL, 0);
            
            printf("Processo pai (PID %d): valor final na memória compartilhada: %d\n", getpid(), *shrPointer);
            
            // Desanexando 
            shmdt(shrPointer);
            
            // Removendo 
            shmctl(segment_id, IPC_RMID, NULL);

        }
    }

    return 0;

}