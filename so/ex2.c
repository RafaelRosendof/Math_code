#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdlib.h>

int n = 10;

typedef struct {
    int len;
    int n_seq[];
} fib;


int main(int argc , char ** argv){

    // buscar o segmento pela key no shmid_ds e caso o elemento não exista, deve criar um novo segmento de memória compartilhada
    int segment_id = shmget(IPC_PRIVATE, sizeof(fib) + n * sizeof(int), S_IRUSR | S_IWUSR);

    if (segment_id == -1) {
        perror("Erro ao criar memória compartilhada");
        return 1;
    }

    pid_t pid_pai = fork();

    if(pid_pai < 0){
        perror("Erro ao iniciar o pai");
        return 1;
    }
    else if(pid_pai == 0){
        //Iniciado o pai agora vamos a função

        //agora estou anexando a memória
        fib *f1 = (fib *)shmat(segment_id , NULL , 0);
        if(f1 == (fib *) -1){
            perror("Filho: erro ao anexar memória ");
            exit(1);
        }

        //represento o N como um array ficando mais fácil de percorrer
        f1 -> len = n;

        if(n >= 1 ) f1 -> n_seq[0] = 0;
        if(n >= 2) f1 -> n_seq[1] = 1;

        for(int i = 2 ; i < n ; i++){
            f1 -> n_seq[i] = f1 -> n_seq[i-1] + f1 -> n_seq[i-2];
        }

        shmdt(f1);

        exit(0);
    }

    else{

        wait(NULL);

        fib * f1 = (fib *) shmat(segment_id , NULL , 0);
        if(f1 == (fib *)- 1){
           perror("Pai -> erro ao anexar a memória ");
          return 1;
        }

        printf("Sequencia de fibonacci com %d termos \n ", f1 -> len);

        for(int i = 0 ; i < f1 -> len ; i++){
            printf("%d " , f1 -> n_seq[i]);
        }
        printf("\n");

        shmdt(f1);
        shmctl(segment_id , IPC_RMID , NULL);
    }
    return 0;
}
