package four

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"

	"gonum.org/v1/gonum/dsp/fourier"
)

func fill_vector(vec []float64) []float64 {

	for i := 0; i < len(vec); i++ {
		num := math.Sin(float64(i*2)) + math.Cos(float64(i*3))
		vec[i] = float64(num)

		vec[i] = vec[i] * rand.Float64()
	}

	return vec
}

func resolve_fft(vec []float64) []complex128 {

	period := float64(len(vec))

	fft := fourier.NewFFT(len(vec))
	coeff := fft.Coefficients(nil, vec)

	for i, c := range coeff {
		fmt.Printf("freq=%v cycles/period, magnitude=%v, phase=%.4g\n",
			fft.Freq(i)*period, cmplx.Abs(c), cmplx.Phase(c))
	}

	return coeff
}

func resolve_dct(vec []float64) []float64 {

	period := float64(len(vec))

	dct := fourier.NewDCT(len(vec))
	coeff := dct.Transform(nil, vec)

	for c := range coeff {
		fmt.Printf("period: %f , coeficente=%f  \n", period, c)
	}

	return coeff
}

func Resolve_dst(vec []float64) []float64 {
	period := float64(len(vec))

	dst := fourier.NewDST(len(vec))
	coeff := dst.Transform(nil, vec)

	for c := range coeff {
		fmt.Printf("period: %f , coeficente=%f  \n", period, c)
	}

	return coeff
}
