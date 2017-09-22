package nn

import "math"

func sigmoid(sum float64) float64  { return 1.0 / (1.0 + math.Exp(-sum)) }
func Dsigmoid(sum float64) float64 { return sum * (1 - sum) }

func tanh(sum float64) float64  { return math.Tanh(sum) }
func Dtanh(sum float64) float64 { return 1.0 - sum*sum }

func activation(sum float64) float64           { return sigmoid(sum) }
func activationDerivative(sum float64) float64 { return Dsigmoid(sum) }
