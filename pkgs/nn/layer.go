package nn

import "strconv"

type Layer struct {
	neurons []*Neuron
	Length  int
}

func NewLayer(nbNeurons int, NBNextConnection int, nbLayer int) *Layer {
	neurons := make([]*Neuron, nbNeurons+1)
	for i := 0; i <= nbNeurons; i++ {
		name := strconv.Itoa(nbLayer) + strconv.Itoa(i)
		neurons[i] = NewNeuron(NBNextConnection, name)
	}
	neurons[nbNeurons].Output = 1
	return &Layer{neurons, nbNeurons}
}

func (l *Layer) Feed(inputs []float64) {
	for i := 0; i < l.Length; i++ {
		l.neurons[i].Output = inputs[i]
	}
}
