#include<stdio.h>


__global__ void hello(void){
    printf("Falando de uma GPU NVIDIA\n");
}

int main(void){
    printf("Falando de uma GPU NVIDIA\n");
 
    hello<<<1,10>>>();

    cudaDeviceReset();

    return 0;
}