#include <iostream>
#include <cmath>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace mp = boost::multiprecision;  


double function(double x) {
    return x * x * x + cos(x);
}



mp::cpp_dec_float_50 bisection(double a, double b, mp::cpp_dec_float_50 error) {

    int reps = 0;
    if (function(a) * function(b) > 0) {
        throw "Intervalo não possui raizes!!!\n";
    }
    int iter = 0; // Initialize iter
    while ((b - a) / 2 > error) {
      double c = (a + b) / 2;
        if (fabs(function(c)) < error) { // Check if c is close to 0
            std::cout << "Achou a raiz !!!";
            return c;
        } else if (function(c) * function(a) < 0) {
            b = c;
        } else {
            a = c;
        }
        std::cout << "iteração " << reps << " E ainda nada !!!!\n";
        reps++;
        iter++; // Increment iter
    }

    return (a + b) / 2;
}

int main() {
    std::cout << "Digite o erro: ";
    mp::cpp_dec_float_50 error;
    std::cin >> error;
    std::cout << "Digite o intervalo: ";
    double a, b;
    std::cin >> a >> b;

    try {
    mp::cpp_dec_float_50 root = bisection(a, b, error);
        std::cout << "Raiz encontrada: " << root << std::endl;
    } catch (const char* msg) {
        std::cerr << "Erro: " << msg << std::endl;
    }

    return 0;
}


/*
#include <iostream>

#include <cmath>

double function( double x){
  return x*x*x + cos(x);
}

double bissection(double a , double b, double erro){
  int reps = 0;
  if(function(a) * function(b) > 0 ){
    throw "Intervalo não possui raizes!!! \n";
  }
  int iter = function(b) - function(a);
  while(erro > iter){
    double c = (a+b)/2;

   // double meio = function(c);

    if(function(c) == 0){
      std::cout<<"Achou a raiz !!!";
      return c; 
    }

    else if(function(c) * function(a) < 0){
      b=c;
    }

    else{
      a=c;
    }
  
    std::cout<<"iteração "<< reps << "E ainda nada !!!!\n";
    reps++;
    iter = function(c);


  }

return -1;
}

int main() {
    std::cout<<"Digite o erro: ";
    double erro;
    std::cin>>erro;
    std::cout << "Digite o intervalo: ";
    double a, b;
    std::cin >> a >> b;

    try {
        double root = bissection(a, b,erro);
        std::cout << "Raiz encontrada: " << root << std::endl;
    } catch (const char* msg) {
        std::cerr << "Erro: " << msg << std::endl;
    }

    return 0;
}
*/
