package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	"os"
	"./reddit"
	"strings"
)

const LEARNING_FACTOR = 0.2

func activation(input float64) float64 {
	return 2.0 / (1.0 + math.Pow(math.E, -input)) - 1.0
}
func activationDerivative(input float64) float64 {
	a := activation(input)
	return a*(1.0-a)
}

type Neuron struct {
	weights []float64
	constant float64
}

func (n Neuron) output(input []float64) float64 {
	total := n.constant
	for i := range n.weights {
		total += n.weights[i] * input[i]
	}
	return activation(total)
}

func (n Neuron) learn(input []float64, errorDerivative float64) []float64 {
	total := n.constant
	for i := range n.weights {
		total += n.weights[i] * input[i];
	}
	derivativeMagnitude := activationDerivative(total)

	for i := range n.weights {
		derivative := derivativeMagnitude * input[i]
		n.weights[i] -= LEARNING_FACTOR * errorDerivative * derivative
	}
	n.constant -= LEARNING_FACTOR * errorDerivative * derivativeMagnitude

	backpropagationError := make([]float64, len(input))
	for i := range input {
		backpropagationError[i] = derivativeMagnitude * n.weights[i] * errorDerivative
	}
	return backpropagationError
}

type Layer struct {
	neurons []Neuron
}

func (l Layer) size() int {
	return len(l.neurons)
}

func (l Layer) output(input []float64) []float64 {
	output := make([]float64, len(l.neurons))
	for i := range l.neurons {
		output[i] = l.neurons[i].output(input)
	}
	return output
}

func (l Layer) learn(input []float64, errorDerivative []float64) []float64 {
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

func makeNetwork(sizes []int, inputSize int) NeuralNetwork {
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

func (n NeuralNetwork) learn(input []float64, expected []float64) {
	outputs := make([][]float64, len(n.layers) + 1)
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
	for i := len(n.layers)-1; i >= 0; i-- {
		backpropagationError = n.layers[i].learn(outputs[i], backpropagationError)
	}
}

func xor() ([]float64, []float64) {
	a := rand.Intn(1000)
	b := rand.Intn(1000)
	output := a ^ b
	return []float64{float64(a)/1000.0, float64(b)/1000.0}, []float64{float64(output)/1000.0}
}

func basicTest() {
	network := makeNetwork([]int{5,5,1},2)
	error := 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.output(input)
		error += (answer[0]-output[0])*(answer[0]-output[0])
	}
	fmt.Println(network)
	fmt.Println("Before:",error)
	for i := 0; i < 1000000; i++ {
		input, output := xor()
		network.learn(input,output)
	}
	error = 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.output(input)
		error += (answer[0]-output[0])*(answer[0]-output[0])
	}
	fmt.Println("After:",error)

	fmt.Println(network)
}

func stringToFloats(data string) []float64 {
//	fmt.Println(data)
//	b64Data := []byte(base64.StdEncoding.EncodeToString([]byte(data)))
//	fmt.Println(string(b64Data))
	input := make([]float64, 100)
	for i, b := range []byte(data) {
		for j := 0; j < 8; j++ {
			input[8*i+j] = float64((b & (1 << j)) >> j)
		}
	}
	return input
}

func intToFloats(data int) []float64 {
	input := make([]float64, 32)
	for j := 0; j < 32; j++ {
		input[j] = float64((data & (1 << j)) >> j)
	}
	return input
}

func redditTest() {
	fmt.Println(" - Starting test")
	comments, err := reddit.ReadComments(os.Args[1], 10000)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(" - Read comments")
	wordSet := make(map[string]int)
	for _, comment := range comments {
		commentWords := strings.Split(comment.Body, " ")
		for _, word := range commentWords {
			if len(word) <= 10 {
				wordSet[word] += 1
			}
		}
	}
	words := make([]string, len(wordSet))
	i := 0
	for word := range wordSet {
		words[i] = word
		i += 1
	}
	fmt.Println(" - Found words")
	inputs := make([][]float64, 0)
	outputs := make([][]float64, 0)
	for i, word := range words {
		if len(word) <= 10 {
			inputs = append(inputs, stringToFloats(word))
			outputs = append(outputs, intToFloats(i))
		}
	}
	fmt.Println(" - Converted to floats")
	network := makeNetwork([]int{50,50,32}, 100)
	trainingDataLimit := len(inputs)*9/10
	for i := 0; i < trainingDataLimit; i++ {
		network.learn(inputs[i], outputs[i])
	}
	fmt.Println(network.output(inputs[trainingDataLimit]), outputs[trainingDataLimit])
	fmt.Println(network)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	redditTest()
	//basicTest()
}
