package neural

import (
	"math"
	"math/rand"
	"sync"
)

type Network interface {
	Output([]float64) []float64
	Learn(Input, Answer)
}

const LEARNING_FACTOR = 0.2
const NUM_THREADS = 16

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
	return math.Max(1.0-a*a, 0.0001)
}

type Input []float64
type Answer []float64

type NetworkData struct {
	Inputs  []Input
	Answers []Answer
}

func MakeNetworkData() NetworkData {
	return NetworkData{
		Inputs:  make([]Input, 0),
		Answers: make([]Answer, 0),
	}
}

func (data *NetworkData) AddData(input Input, answer Answer) {
	data.Inputs = append(data.Inputs, input)
	data.Answers = append(data.Answers, answer)
}

func (data *NetworkData) Split(n int) (NetworkData, NetworkData) {
	inputs1 := data.Inputs[:n]
	inputs2 := data.Inputs[n:]
	answers1 := data.Answers[:n]
	answers2 := data.Answers[n:]
	data1 := NetworkData{Inputs: inputs1, Answers: answers1}
	data2 := NetworkData{Inputs: inputs2, Answers: answers2}
	return data1, data2
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

func (n *NeuralNetwork) Learn(input Input, expected Answer) {
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

func MakeNetwork(sizes []int, inputSize int) *NeuralNetwork {
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
	return &NeuralNetwork{
		layers: layers,
	}
}

type FastNeuralNetwork struct {
	sizes   []int
	weights [][][]float64
	biases  [][]float64
}

func NewFastNeuralNetwork(dimensions []int, inputSize int) *FastNeuralNetwork {
	sizes := make([]int, len(dimensions))
	copy(sizes, dimensions)
	weights := make([][][]float64, len(sizes))
	biases := make([][]float64, len(sizes))
	for layer := 0; layer < len(weights); layer++ {
		weights[layer] = make([][]float64, sizes[layer])
		biases[layer] = make([]float64, sizes[layer])
		for neuron := 0; neuron < len(weights[layer]); neuron++ {
			previousLayerSize := inputSize
			if layer > 0 {
				previousLayerSize = sizes[layer-1]
			}
			weights[layer][neuron] = make([]float64, previousLayerSize)
			for k := 0; k < len(weights[layer][neuron]); k++ {
				weights[layer][neuron][k] = rand.Float64()
			}
			biases[layer][neuron] = rand.Float64()
		}
	}
	return &FastNeuralNetwork{sizes: sizes, weights: weights, biases: biases}
}

type pair struct {
	layer, neuron int
}

func calculateNeuron(outputs [][]float64, network *FastNeuralNetwork, instructions chan pair, wg *sync.WaitGroup, finished *sync.WaitGroup) {
	for {
		i, ok := <-instructions
		if !ok {
			break
		}
		inputs := outputs[i.layer]
		weights := network.weights[i.layer][i.neuron]
		sum := network.biases[i.layer][i.neuron]
		for k := range weights {
			sum += weights[k] * inputs[k]
		}
		outputs[1+i.layer][i.neuron] = tanh(sum)
		wg.Done()
	}
	finished.Done()
}

func (n *FastNeuralNetwork) Output(input []float64) []float64 {
	// var wg sync.WaitGroup
	// var finished sync.WaitGroup
	// instructions := make(chan pair, NUM_THREADS+1)

	// outputs := make([][]float64, len(n.sizes)+1)
	// outputs[0] = make([]float64, len(input))
	// copy(outputs[0], input)

	// finished.Add(NUM_THREADS)
	// for i := 0; i < NUM_THREADS; i++ {
	// 	go calculateNeuron(outputs, n, instructions, &wg, &finished)
	// }

	// for i := range outputs {
	// 	if i == 0 {
	// 		continue
	// 	}
	// 	layer := i - 1
	// 	outputs[i] = make([]float64, n.sizes[layer])
	// 	wg.Add(len(outputs[i]))
	// 	for neuron := 0; neuron < len(outputs[i]); neuron++ {
	// 		instructions <- pair{layer: layer, neuron: neuron}
	// 	}
	// 	wg.Wait()
	// }
	// close(instructions)
	// finished.Wait()
	// return outputs[len(outputs)-1]

	outputs := make([][]float64, len(n.sizes))
	for layer := range outputs {
		// next := make([]float64, n.sizes[layer])
		// layerWeights := n.weights[layer]
		// for k := 0; k < len(input); k++ {
		// 	temp := input[k]
		// 	for neuron := range next {
		// 		next[neuron] += layerWeights[neuron][k] * temp
		// 	}
		// }
		// for i := 0; i < len(next); i++ {
		// 	next[i] = tanh(next[i])
		// }
		// outputs[layer] = next
		// input = next

		outputs[layer] = make([]float64, n.sizes[layer])
		layerWeights := n.weights[layer]
		layerBiases := n.biases[layer]
		for neuron := range outputs[layer] {
			weights := layerWeights[neuron]
			sum := layerBiases[neuron]
			for k := 0; k < len(weights); k++ {
				sum += weights[k] * input[k]
			}
			outputs[layer][neuron] = tanh(sum)
		}
		input = outputs[layer]
	}
	return outputs[len(outputs)-1]
}

func (n *FastNeuralNetwork) Learn(input Input, expected Answer) {

}
