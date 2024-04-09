#include <iostream>
#include<cmath>
 
#include<boost/multiprecision/cpp_dec_float.hpp>

namespace mp = boost::multiprecision;
            
double func(double x){
  return x*x - sin(x);
}

double derivada(double x){
  return 2*x - cos(x);
}

void newton(double x, mp::cpp_dec_float_50 erro ){

  double h = func(x) / derivada(x);
  int iter =0;
  while(fabs(h) > erro){
    h = func(x)/derivada(x);

    std::cout<<"Iteração número: "<<iter <<"Com o erro: "<< h << std::endl;

    //fazendo x+1 
    x = x - h;
    iter++;
  }

  std::cout<< "Valor da raiz é :" << x;

}

int main(){
  
  std::cout<<"Digite o erro: ";
  mp::cpp_dec_float_50 erro;
  std::cin>>erro;

  //std::cout<<"\n Digite os intervalos: ";
  //double a ,b;
  //std::cin>>a >>b;
  std::cout<<"\n\n Digite o X: ";
  double x;
  std::cin>>x;

  newton(x , erro);
  return 0;
}
