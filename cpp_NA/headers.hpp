#ifndef HEADERS_H
#define HEADERS_H

// Defining the libs and includes
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <functional>

// Create the methods

/*
Linear System{
    LU
    Gauss
    Cholesky

    Jacobi-Richardson
    Gauss-Siedel

    Matrix Dot
    Matrix Vecdor Dot
}
*/

class Sistemas_Lineares
{
public:
    using Matrix = std::vector<std::vector<double>>;
    using Vector = std::vector<double>;

    bool square(const Matrix mat);

    Vector LU_dec(const Matrix &matA, const Vector &vecB);

    Vector Gauss(const Matrix &matA, const Vector &vecB);

    Vector Cholesky(const Matrix& matA , const Vector &vecB);

    Vector JacobiRichardson(const Matrix& matA , const Vector &vecB , const double erro , int MaxIter);
    
    Vector Gauss_Siedel(const Matrix &matA , const Vector &vecB , const double erro , int MaxIter);

    Matrix Dot(const Matrix &matA , const Matrix &matB);

    
};

/*
Calculus{
    Integration
    Newton-Cotes
    Gauss quads{
        Gauss-Legendre
        Gauss-Tchebyshev
        Gauss-Laguerre
        Gauss-Hermite
    }
    Romberg Integration
    Trapezoidal rulde
    Simpson's Rules 

    Differentiation{
        High accuracy Differentiation
        Richardson Extrapolation
        Partial Derivatives
    }

    Otimization 
}
*/

class Calculus{
    public:
    //this is functional methods for passing the functions to the calculus methods
    double NewtonCotes(std::function<double(double)> f, double a , double b , int n);
    double RombergIntegration(std::function<double(double)> f, double a , double b , int max_steps, double tol);
    double TrapezoidalRule(std::function<double(double)> f, double a , double b , int n , double tolerance);
    double SimpsonsRule38(std::function<double(double)> f, double a , double b , int n);
    double SimpsonsRule13(std::function<double(double)> f, double a , double b , int n);
    double HighAccuracyDifferentiation(std::function<double(double)> f, double x , double h);
    double RichardsonExtrapolation(std::function<double(double)> f, double x , double h , int n);
    double PartialDerivatives(std::function<double(double, double)> f, double x, double y, int var, double h);
    double booleMethod(std::function<double(double)> f, double a , double b , int n);
    double AdaptiveQuadrature(std::function<double(double)> f, double a , double b , double tol);
    double MonteCarloIntegration(std::function<double(double)> f, double a , double b , int n);
    double 
    //double MeanSquare(); //Otimization
    //otimization: TODO 
};



//class Calculus Todo()!

/*
class Algebra{
    //Lagrange 
    //Lagrange equal spaces

    //Linear interpolation

    //Newton 
    //Newton Gregory

    //regula falsi
    //newton
    //bissection
    //iterative
    //secant 
    

}
*/
class Algebra{
    public:
    //Interpolation

    //std::function to

    double Lagrange();
    double LagrangeEqual();
    double Newton();
    double NewtonGregory();

    //find roots
    //here the std::functional to 
    double regulaFalsi();
    double bissection();
    double iterative();
    double secant();
    double newtonRaphson();

};

/*

EiginValues{
    Power Method
    Inverse Power Method
    QR Method
    Jacobi Method
    Rayleigh Quotient
}

autovalores: TODO

ODE{
    Euler
    Runge-Kutta
    Adams-Bashforth
    Adams-Moulton
    Backward Differentiation
    Predictor-Corrector
    Finite Difference
    Finite Element
    Boundary Element
    Spectral Methods
    Galerkin
    Collocation
    Least Squares
    Finite Volume
    }

ODE: TODO

PDE{
    Finite Difference
    Finite Element
    Boundary Element
    Spectral Methods
    Galerkin
    Collocation
    Least Squares
    Finite Volume
}

PDE: TODO

Fourrier transform{
    FFT
    DFT
    IDFT
    FFT2D
    DFT2D
    IDFT2D
}


*/

#endif