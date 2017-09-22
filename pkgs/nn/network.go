package nn

import "fmt"

import "math"

const (
	eta   = 0.15
	alpha = 0.5
)

type Network struct {
	layers          []*Layer
	err             float64
	recentAvg       float64
	recentAvgFactor float64
}

func New(topology []int) *Network {
	length := len(topology)
	layers := make([]*Layer, length)
	for i := 0; i < length; i++ {
		if i == length-1 {
			layers[i] = NewLayer(topology[i], 0, i)

		} else {
			layers[i] = NewLayer(topology[i], topology[i+1], i)
		}
	}
	return &Network{layers, 0.0, 0.0, 0.0}
}

func (n *Network) FeedForward(inputs []float64) {
	if len(inputs) != n.layers[0].Length {
		fmt.Println("Failed ", n.layers[4])
		return
	}
	n.layers[0].Feed(inputs)
	length := len(n.layers)
	for i := 1; i < length; i++ {
		prev := n.layers[i-1]
		for j := 0; j < n.layers[i].Length; j++ {
			n.layers[i].neurons[j].FeedForward(prev, j)
		}
	}
}

func (n *Network) computeAvgError() {
	n.recentAvg = (n.recentAvg*n.recentAvgFactor + n.err) / (n.recentAvgFactor + 1.0)
}

func (n *Network) computeErr(expected []float64) {
	length := len(n.layers)
	layer := n.layers[length-1]
	err := 0.0
	for i := 0; i < layer.Length; i++ {
		delta := expected[i] - layer.neurons[i].Output
		err += delta * delta
	}
	err /= float64(length - 1)
	n.err = math.Sqrt(err)
}

func (n *Network) computeOutputLayerGradients(expected []float64) {
	length := len(n.layers)
	layer := n.layers[length-1]
	for i := 0; i < layer.Length; i++ {
		layer.neurons[i].computeOutputGradients(expected[i])
	}
}

func (n *Network) computeHiddenLayerGradients(expected []float64) {
	for i := len(n.layers) - 2; i > 0; i-- {
		for j := 0; j < n.layers[i].Length+1; j++ {
			n.layers[i].neurons[j].computeHiddenGradients(n.layers[i+1], j)
		}
	}
}

func (n *Network) updateConnectionWeight() {
	for i := len(n.layers) - 1; i > 0; i-- {
		for j := 0; j < n.layers[i].Length; j++ {
			n.layers[i].neurons[j].updateConnectionWeight(n.layers[i-1], j)
		}
	}
}

func (n *Network) BackProb(expected []float64) {
	n.computeErr(expected)
	n.computeOutputLayerGradients(expected)
	n.computeHiddenLayerGradients(expected)
	n.updateConnectionWeight()
}

func (n *Network) GetResults() []float64 {
	lastLayer := n.layers[len(n.layers)-1]
	ret := make([]float64, lastLayer.Length)
	for i := 0; i < lastLayer.Length; i++ {
		ret[i] = lastLayer.neurons[i].Output
	}
	return ret
}
