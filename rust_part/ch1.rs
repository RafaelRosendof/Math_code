/*

Methods for latter study on rust

ODE/PDE solvers in pure rust

implement Runge-Kutta methods (RK4 , Dormand-Prince) for ODEs
implicit solvers for PDEs BDF
Euler
Forward Euler and others

Resolving and simulate the heat, wave and burgers equations
Forward Euler and others

For the fft part

1d and 2d fft from scratch and use the rustfft lib

implement the PDS method for the spectogram

solve the heat and burgues equation the wave to

try to plot some of the PDS and fft methods


For the numerical part

Some Linear algebra methods maybe using nalgebra or ndarray
Some bindings to linear algebra libraries like LAPACK or BLAS  bindgen or cxx

try to create some 3d Plot to visualize the results

the part of linear systemns and eigenvalues

the part of integration and differentiation

ndarray: For N-dimensional arrays, NumPy-like.

nalgebra: For algebraic structures and matrix/vector operations.

rustfft: FFT implementation.

plotters or gnuplot: For plotting numerical results.

rayon: Data-parallelism (for vectorized loops).

criterion: For benchmarking solvers.


for the first steps we gonna implement{
    trapezoidal integral
    simpson integral 1/3
    simpson integral 3/8
    numerical differentiation
    Forward differentiation
    Backward differentiation
    matrix multiplication
    matrix addition
    matrix subtraction
}

*/

fn main() {
    println!("Hello, world!");

    let mut x = 5;
    for i in 1..=10 {
        println!("The value of i is: {}", i);
        x += 1;
    }

    println!("The value of x is: {}", x);
}
