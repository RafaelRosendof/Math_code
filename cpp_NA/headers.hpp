#ifndef HEADERS_H
#define HEADERS_H

// Defining the libs and includes
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstddef>

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
}
*/

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

};

#endif