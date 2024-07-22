#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>


//lambda expression
auto f = [](double x) {return std::log10(std::sqrt(x*x + x) + 1); };




//trapezoidal rule

double trapezoidal(double a , double b , std::function<double(double)> f , int n){
    double h = (b-a)/n;

    double integral = (f(a) + f(b)) / 2.0;

    #pragma omp parallel for reduction(+:integral) 
    for(int i = 1 ; i < n ;++i){

        //#pragma omp critical
        //{
        //    std::cout << "Thread " << omp_get_thread_num() << " is processing element " << i << std::endl;
       // }

        integral += f(a + i*h);
    }
    integral *= h;

    return integral;
}


double erro_trapezoidal(std::function<double(double)> f , double a , double b , double tol){
    int n = 1;
    double integral_prev = trapezoidal(a , b , f , n);

    //#pragma omp parallel while reduction(+:integral)
    while(true){
        n*=2;

        double integral = trapezoidal(a , b , f , n);

        if(std::abs(integral - integral_prev) < tol){
            return integral;
        }
        integral_prev = integral; //atualiza 
    }
}


//simpson 3/8 

double simpson_38(double a , double b , std::function<double(double)> f , int n){
    if(n % 3 != 0){
        throw std::invalid_argument("n must be a multiple of 3");
    }

    double h = (b-a)/n;

    double integral = f(a) + f(b);
    #pragma omp parallel for reduction(+:integral)
    for(int i = 1 ; i < n ; ++i){
        if(i % 3 == 0){
            integral += 2*f(a + i*h);
        }else{
            integral += 3*f(a + i*h);
        }
    }

    integral = 3 * h * integral/8; 

    return integral;
}

int main(){
    double a = 2;
    double b = 100;
    int n = 60;
    double tol = 1e-11;
    int precision = 15;

    auto res = simpson_38(a , b , f , n);
    std::cout << "Integral (Simpson 3/8): " << std::setprecision(precision) << res << std::endl;

    std::cout<< "\n\n\n\n";

    auto res2 = erro_trapezoidal(f , a , b , tol);

    std::cout<< "Trapezoidal Rule\n" << std::setprecision(precision) << res2 << std::endl;
}

/*
Boole method
Adaptive Quadrature
Monte carlo integration
Romberg Integration
Newton Cotes 
*/