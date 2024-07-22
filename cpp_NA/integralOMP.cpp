#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>

double f(double x) {
    return log10(sqrt(x*x + x) + 1);
}

double trapezoidal_rule_omp(double a, double b, int n) {
    double h = (b - a) / n;
    double integral = (f(a) + f(b)) / 2.0;

    #pragma omp parallel for reduction(+:integral)
    for (int i = 1; i < n; ++i) {
        integral += f(a + i * h);
    }

    integral *= h;
    return integral;
}

int main() {
    double a = 2;
    double b = 100;
    int n = 1000000; // Aumentar o número de subintervalos para maior precisão

    double integral = trapezoidal_rule_omp(a, b, n);
    
    std::cout << "Integral (OpenMP Trapézio): " << std::setprecision(30) << integral << std::endl;

    return 0;
}
