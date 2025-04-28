#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>


#define TOLERANCIA 1e-15
double fun(x){
    return x*x + sqrt(x) * exp(42*x) * sin(10*x) + 1;
}

int trapezio(double (*f)(double), double a, double b, int n){
    int h = (b - a) / n;

    double integral = (f(a) - f(b)) / 2.0;

    for(int i = 0 ; i < n; i++){
        integral += f(a + i * h);
    }
    integral *= h;

    return integral;
}

void calculo(double x){
    int n = 1000000;
    double error = 1;
    int prev_res = 0;
    int iter = 0;
    double res = 0;
    while(error > TOLERANCIA){
        res = trapezio(fun, 0, x, n);
        error = fabs(res - prev_res);
        prev_res = res;
        n *= 2;
        iter++;
        //printf("Iteração atual %d: %f\n", iter, res);
    }
    printf("PID %d: Resultado final = %f (Iterações: %d)\n", getpid(), res, iter);

   // return res , n , iter;
}

int main(int argc , char **arvg){

    pid_t pid;
    int status;
    int num_forks = 4;

    printf("Processor pai %d", getpid());

    for(int i = 0 ; i < num_forks ; i++){
        pid = fork();
        if(pid < 0){
            perror("fork foi de f");
            exit(1);
        }
        else if (pid == 0 ){
            printf("Processo filho %d, PID %d\n", i, getpid());
            calculo(42); //calculand
            exit(0);
        }
    }

    for(int i = 0 ; i < num_forks ; i++){
        pid_t filhoPid = wait(&status);
        printf("Pai -> Filho com PID %d foi finalizado    ::::  " , filhoPid);
    }

    printf("Por fim todos finalizados");

    return 0;
}