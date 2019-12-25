package main

import (
	"fmt"
	"math"
)

func activation(input float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -input))
}

type Neuron struct {
	weights []float64
}

func (n Neuron) output(input []float64) float64 {
	total := 0.0
	for i := range n.weights {
		total += n.weights[i] * input[i]
	}
	return activation(total)
}

type Layer struct {
	neurons []Neuron
}

func (l Layer) output(input []float64) []float64 {
	output := make([]float64, len(l.neurons))
	for i := range l.neurons {
		output[i] = l.neurons[i].output(input)
	}
	return output
}

type NeuralNetwork struct {
	layers []Layer
}

func makeNetwork(sizes []int64, inputSize int64) NeuralNetwork {
	layers := make([]Layer, len(sizes))
	for i := range layers {
		neurons := make([]Neuron, sizes[i])
		for j := range neurons {
			var weights []float64
			if j == 0 {
				weights = make([]float64, inputSize)
			} else {
				weights = make([]float64, sizes[j-1])
			}
			for k := range weights {
				weights[k] = 1.0
			}
			neurons[j] = Neuron{
				weights: weights,
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

func (n NeuralNetwork) output(input []float64) []float64 {
	values := input
	for _, layer := range n.layers {
		values = layer.output(values)
	}
	return values
}

func main() {
	layer1 := Layer{neurons: []Neuron{Neuron{weights: []float64{1.0, 2.0, 3.0}}, Neuron{weights: []float64{-1.0, 2.0, 0.1}}}}
	layer2 := Layer{neurons: []Neuron{Neuron{weights: []float64{2.0, 2.0}}, Neuron{weights: []float64{5.0, -3.0}}}}
	network := NeuralNetwork{layers: []Layer{layer1, layer2}}
	fmt.Println(network.output([]float64{2.0, 0.1, 0.3}))
}
