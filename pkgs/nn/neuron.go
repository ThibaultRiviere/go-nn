package nn

import "math/rand"

func randomWeight() float64 { return rand.Float64() }

type Connection struct {
	weight      float64
	deltaWeight float64
}

func NewConnection() *Connection {
	return &Connection{randomWeight(), 0}
}

type Neuron struct {
	Output      float64
	connections []*Connection
	gradient    float64
	name        string
}

func NewNeuron(NBNextConnection int, name string) *Neuron {
	connections := make([]*Connection, NBNextConnection)
	for i := 0; i < NBNextConnection; i++ {
		connections[i] = NewConnection()
	}
	return &Neuron{0, connections, 0.0, name}
}

func (n *Neuron) FeedForward(prevLayer *Layer, index int) {
	sum := float64(0.0)
	for _, neuron := range (*prevLayer).neurons {
		add := neuron.Output * neuron.connections[index].weight
		sum += add
	}
	n.Output = activation(sum)
}

func (n *Neuron) updateConnectionWeight(prevLayer *Layer, index int) {
	for i := 0; i < prevLayer.Length+1; i++ {
		oldDelta := prevLayer.neurons[i].connections[index].deltaWeight
		newDelta := eta*prevLayer.neurons[i].Output*n.gradient + alpha*oldDelta
		prevLayer.neurons[i].connections[index].deltaWeight = newDelta
		prevLayer.neurons[i].connections[index].weight += newDelta
	}
}

// Compute the affect of the neuron error on the next layer.
func (n *Neuron) sumErrLayer(nextLayer *Layer) float64 {
	sum := 0.0
	for i := 0; i < nextLayer.Length; i++ {
		sum += n.connections[i].weight * nextLayer.neurons[i].gradient
	}
	return sum
}

func (n *Neuron) computeHiddenGradients(layer *Layer, index int) {
	dow := n.sumErrLayer(layer)
	derivation := activationDerivative(n.Output)
	n.gradient = dow * derivation
}

func (n *Neuron) computeOutputGradients(expected float64) {
	delta := expected - n.Output
	n.gradient = delta * activationDerivative(n.Output)
}
