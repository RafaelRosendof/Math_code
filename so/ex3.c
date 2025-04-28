#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdlib.h>



int n = 10;

int main(int argc , char **argv){
    //criado
    int seg_id = shmget(IPC_PRIVATE , sizeof(int) , S_IRUSR | S_IWUSR);

    //anexado segmento
    int * shr_sum = (int *)shmat(seg_id , NULL , 0);

    for(int i = 0 ; i <= n ; i++){
        pid_t pid = fork();

        if(pid == 0){
            *shr_sum +=i;

            //free
            shmdt(shr_sum);
            exit(0);
        }else if(pid > 0){
            wait(NULL);

            if (i == n){
                //último
                printf("A soma dos primeiros %d números naturais é: %d\n", n, *shr_sum);
            }
        }
        else{
            perror("Falhou no primeiro fork");
            exit(1);
        }
    }

    shmdt(shr_sum);
    shmctl(seg_id , IPC_RMID , NULL);

    return 0;
}
