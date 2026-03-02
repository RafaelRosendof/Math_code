package integral

import (
	//"gonum.org/v1/gonum/mat"
	"fmt"

	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/integrate/quad"
)

func Test_simple_integral(x_asis []float64, y_axis []float64) float64 {

	fmt.Println("Testing the gonum integration")

	//mat.Matrix
	sum := integrate.Trapezoidal(x_asis, y_axis)
	//fmt.Printf("Returning the sum bellow the curve %2.f", sum)

	return sum
}

func Test_integral_simpson(x_asis []float64, y_axis []float64) float64 {

	//fmt.Println("Testing the gonum simpson integration")

	sum := integrate.Simpsons(x_asis, y_axis)

	//fmt.Printf("Returning the sum bellow the curve %2.f", sum)

	return sum
}

func Test_integral_romberg(y_axis []float64) float64 {
	//fmt.Println("Testing the gonum romberg integration")

	sum := integrate.Romberg(y_axis, 1)
	sum2 := integrate.Romberg(y_axis, 5)
	sum3 := integrate.Romberg(y_axis, 10)

	fmt.Printf("Show the values to a domain 1 -> %2.f\n", sum)
	fmt.Printf("\n\nShow the values to a domain 5 -> %2.f\n", sum2)
	fmt.Printf("\n\nShow the values to a domain 10 -> %2.f\n", sum3)

	//fmt.Printf("Returning the sum bellow the curve %2.f", sum)

	return sum
}

func Test_the_quads(f func(float64) float64, a float64, b float64, n int, concurrency int) float64 {

	return quad.Fixed(f, a, b, n, nil, concurrency)
}
