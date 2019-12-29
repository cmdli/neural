package neural

import (
	"math"
	"math/rand"
)

const LEARNING_FACTOR = 0.2

func sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -input))
}
func sigmoidDerivative(input float64) float64 {
	a := sigmoid(input)
	return a * (1.0 - a)
}

func tanh(input float64) float64 {
	return 2.0*sigmoid(2.0*input) - 1.0
}
func tanhDerivative(input float64) float64 {
	a := tanh(input)
	return 1.0 - a*a
}

type Neuron struct {
	weights  []float64
	constant float64
}

func (n Neuron) output(input []float64) float64 {
	total := n.constant
	for i := range n.weights {
		total += n.weights[i] * input[i]
	}
	return tanh(total)
}

func (n *Neuron) learn(input []float64, errorDerivative float64) []float64 {
	total := n.constant
	for i := range n.weights {
		total += n.weights[i] * input[i]
	}
	derivativeMagnitude := tanhDerivative(total)

	for i := range n.weights {
		derivative := derivativeMagnitude * input[i]
		n.weights[i] -= LEARNING_FACTOR * errorDerivative * derivative
	}
	n.constant -= (LEARNING_FACTOR * errorDerivative * derivativeMagnitude)

	backpropagationError := make([]float64, len(input))
	for i := range input {
		backpropagationError[i] = derivativeMagnitude * n.weights[i] * errorDerivative
	}
	return backpropagationError
}

type Layer struct {
	neurons []Neuron
}

func (l *Layer) size() int {
	return len(l.neurons)
}

func (l *Layer) output(input []float64) []float64 {
	output := make([]float64, len(l.neurons))
	for i := range l.neurons {
		output[i] = l.neurons[i].output(input)
	}
	return output
}

func (l *Layer) learn(input []float64, errorDerivative []float64) []float64 {
	backpropagationError := make([]float64, len(input))
	for i := range l.neurons {
		neuronBackpropagationError := l.neurons[i].learn(input, errorDerivative[i])
		for j := range backpropagationError {
			backpropagationError[j] += neuronBackpropagationError[j]
		}
	}
	return backpropagationError
}

type NeuralNetwork struct {
	layers []Layer
}

func (n *NeuralNetwork) Output(input []float64) []float64 {
	values := input
	for _, layer := range n.layers {
		values = layer.output(values)
	}
	return values
}

func (n *NeuralNetwork) Learn(input []float64, expected []float64) {
	outputs := make([][]float64, len(n.layers)+1)
	outputs[0] = make([]float64, len(input))
	for i := range n.layers {
		outputs[i+1] = make([]float64, n.layers[i].size())
	}
	copy(outputs[0], input)
	for i := range n.layers {
		copy(outputs[i+1], n.layers[i].output(outputs[i]))
	}
	backpropagationError := make([]float64, len(expected))
	lastOutput := outputs[len(outputs)-1]
	for i := range expected {
		backpropagationError[i] = (lastOutput[i] - expected[i])
	}
	for i := len(n.layers) - 1; i >= 0; i-- {
		backpropagationError = n.layers[i].learn(outputs[i], backpropagationError)
	}
}

func MakeNetwork(sizes []int, inputSize int) NeuralNetwork {
	layers := make([]Layer, len(sizes))
	for i := range layers {
		neurons := make([]Neuron, sizes[i])
		for j := range neurons {
			var weights []float64
			if i == 0 {
				weights = make([]float64, inputSize)
			} else {
				weights = make([]float64, sizes[i-1])
			}
			for k := range weights {
				weights[k] = rand.Float64()
			}
			neurons[j] = Neuron{
				weights:  weights,
				constant: rand.Float64()*2.0 - 1.0,
			}
		}
		layers[i] = Layer{
			neurons: neurons,
		}
	}
	return NeuralNetwork{
		layers: layers,
	}
}
