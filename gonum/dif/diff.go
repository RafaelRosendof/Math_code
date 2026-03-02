package dif

import (
	"fmt"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

func Example_derivative(f func(float64) float64, x float64) {
	fmt.Println("Printing the derivate of the function F")

	df := fd.Derivative(f, x, &fd.Settings{
		Formula:     fd.Forward,
		Step:        1e-4,
		Concurrent:  true, // False
		OriginKnown: true, // False
		OriginValue: f(0),
	})

	fmt.Printf("f′(%.2f) ≈ %v\n", x, df)
}

func Example_gradient(f func([]float64) float64, x []float64) []float64 {
	fmt.Println("Printing the gradient of the function F in all variables")

	grad := make([]float64, len(x))

	fd.Gradient(grad, f, x, &fd.Settings{
		Formula:     fd.Central,
		Concurrent:  true, //false,
		OriginKnown: true, //false,
		OriginValue: f(x),
	})

	fmt.Println("∇F(x) ≈", grad)

	return grad
}

func Example_hessian(f func([]float64) float64, x []float64) *mat.SymDense {
	fmt.Println("Printing the hessian of the function F in all variables")

	n := len(x)

	hessian := mat.NewSymDense(n, nil)

	fd.Hessian(hessian, f, x, &fd.Settings{
		Formula:    fd.Central,
		Concurrent: true, //false,
	})

	fmt.Printf("H(x) ≈\n%v\n", mat.Formatted(hessian, mat.Prefix("    ")))

	return hessian
}

func Example_jacobian(f func(dst, x []float64), x []float64, m int) *mat.Dense {
	fmt.Println("Printing the jacobian of the function F in all variables")

	n := len(x)
	jac := mat.NewDense(m, n, nil)

	fd.Jacobian(jac, f, x, &fd.JacobianSettings{
		Formula:    fd.Central,
		Concurrent: true, //false,
		Step:       1e-8,
	})

	fmt.Printf("J(x) ≈\n%v\n", mat.Formatted(jac, mat.Prefix("    ")))

	return jac
}

func Example_laplacian(f func([]float64) float64, x []float64) float64 {
	fmt.Println("Printing the laplacian of the function F")

	laplacian := fd.Laplacian(f, x, &fd.Settings{
		Formula:     fd.Central2nd,
		Concurrent:  false, //false,
		OriginKnown: true,  //false,
		//OriginValue: f(x),
	})

	fmt.Printf("∆F(x) ≈ %v\n", laplacian)
	return laplacian
}
