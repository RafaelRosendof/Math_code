package economy

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Minimal_squares(yData []float64) {

	xData := make([]float64, len(yData))

	for i := 0; i < len(yData); i++ {
		xData[i] = float64(i) // days
	}

	n := len(xData)

	A := mat.NewDense(n, 2, nil)
	b := mat.NewVecDense(n, yData)

	for i := 0; i < n; i++ {
		A.Set(i, 0, xData[i])
		A.Set(i, 1, 1.0)
	}

	x := mat.NewVecDense(2, nil)

	err := x.SolveVec(A, b)

	if err != nil {
		fmt.Printf("SolveVec failed: %v\n", err)
		return
	}

	fmt.Printf("Estimated coefficients [slope, intercept]:\n%.4v\n", mat.Formatted(x))
}
