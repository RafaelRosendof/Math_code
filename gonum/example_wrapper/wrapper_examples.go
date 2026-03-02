package examplewrapper

import (
	"fmt"
	"math"
	"math/rand"
	"math_test/dif"
	"math_test/integral"

	"gonum.org/v1/gonum/stat/distuv"
)

func Derivate_examples() {

	fmt.Println("Tesinting all derivates")

	f := func(x float64) float64 {
		return x * (math.Cos(x*x*x) + math.Log10(x*123*x) - math.Pow(x, math.Log2(x*x*x)))
	}

	f_multi := func(x []float64) float64 {
		x0, x1, x2, x3 := x[0], x[1], x[2], x[3]

		term1 := math.Exp(-(x0*x0 + x1*x1)) * math.Cos(x0*x1*x2)

		innerLog := math.Abs(x2*123*x3) + 1e-9
		term2 := math.Log10(innerLog) * math.Pow(math.Abs(x0), math.Log2(math.Abs(x1*x2*x3)+1e-9))

		term3 := math.Sin(math.Pow(x3, 15)) / (1 + x0*x0 + x1*x1 + x2*x2)

		return term1 + term2 - term3
	}

	fmt.Println("Printing the derivate of the function F as the basic derivation")
	dif.Example_derivative(f, math.Pi/2)

	fmt.Println("Printing the gradient of the function F in all variables")

	point := []float64{1.3, 0.2, 2.9, 0.9}

	grad := dif.Example_gradient(f_multi, point)

	fmt.Println("Printing the gradient result", grad)

	fmt.Println("Printing the hessian of the function F in all variables")

	hess := dif.Example_hessian(f_multi, point)

	fmt.Println("Printing the hessian result", hess)

	fmt.Println("Printing the jacobian of the function F in all variables")

	f_vec := func(dst, x []float64) {
		x0, x1, x2, x3 := x[0], x[1], x[2], x[3]

		dst[0] = math.Sin(x0*x1)*math.Exp(-x2) + math.Cos(x3)
		dst[1] = math.Log1p(math.Abs(x0)) * math.Pow(math.Abs(x1), math.Sin(x2))
		dst[2] = (math.Sqrt(math.Abs(x0*x3)) + 1) / (1 + x1*x1 + x2*x2)
		dst[3] = x0 * (math.Cos(x1*x2*x3) + math.Log10(math.Abs(x0*123)+1) - math.Pow(math.Abs(x1), 0.5))

	}

	point_vec := []float64{1.0, 2.0, 0.5, 1.5}
	res := make([]float64, 4)
	f_vec(res, point_vec)

	res_jacob := dif.Example_jacobian(f_vec, point_vec, 4)

	fmt.Println("Printing the jacobian result", res_jacob)

	fmt.Println("Printing the laplacian of the function F")

	laplacian := dif.Example_laplacian(f_multi, point)

	fmt.Println("Printing the laplacian result", laplacian)

}

func Integration_example() {
	points := 1_000_000
	romb := math.Pow(2, 21) + 1

	f1 := func(x float64) float64 {
		term1 := math.Log10(math.Sqrt(x*x*x*10) + math.Cos(10*math.Pow(x, 5)+math.Pi/10))
		term2 := math.Sin(3*x*x) * math.Pow(math.E*150, math.Pow(x, 10))
		term3 := 150 * x * x * math.Log10(math.Pow(x, 5)+math.Cos(20*math.Pow(x, 4)))

		return ((term3 / term1) + term2*float64(rand.Intn(5000)))
	}

	//f2 := func(x float64) float64 {
	//	//d := distuv.Poisson{Lambda: 100}
	//	d := distuv.Exponential{Rate: 10, Src: nil}
	//	return float64(d.Prob(x))
	//}

	f3 := func(x float64) float64 {
		d := distuv.Gamma{Alpha: 0.8, Beta: 0.5, Src: nil}
		return float64(d.Prob(x))
	}

	x_axis := make([]float64, points)
	y_axis := make([]float64, points)
	x_romb := make([]float64, int(romb))
	y_romb := make([]float64, int(romb))

	for i := 0; i < int(len(x_axis)); i++ {
		x_axis[i] = float64((0.1 / float64(points-1) * float64(i)) + 1e-7)
		y_axis[i] = f1(float64(x_axis[i]))
	}

	for i := 0; i < int(len(x_romb)); i++ {
		x_romb[i] = float64((0.1 / float64(points-1) * float64(i)) + 1e-7)
		y_romb[i] = f1(float64(x_romb[i]))
	}

	fmt.Println(x_axis[0], x_axis[1])

	fmt.Println("Starting the tests on the integration field")

	res1 := integral.Test_simple_integral(x_axis, y_axis)

	fmt.Printf("\n\nReturning the sum bellow the curve with the trapezoidal method %.15f\n", res1)

	res2 := integral.Test_integral_simpson(x_axis, y_axis)
	fmt.Printf("\n\nReturning the sum bellow the curve with the simpson method %.15f\n", res2)

	res3 := integral.Test_integral_romberg(y_romb)
	fmt.Printf("\n\nReturning the sum bellow the curve with the romberg method %.15f", res3)

	res4 := integral.Test_the_quads(f3, 1e-8, float64(1), 10000, 5)

	fmt.Printf("\n\nReturning the sum bellow the curve with the fixed method %.15f", res4)

}

func Interpolation_example() {

}
