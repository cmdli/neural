package main

import (
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"strings"
	"time"

	"./neural"
	"./reddit"
)

var alphabet_filter *regexp.Regexp

func initRegex() {
	alphabet_filter, _ = regexp.Compile("[A-Za-z]")
}

func xor() ([]float64, []float64) {
	a := rand.Intn(1000)
	b := rand.Intn(1000)
	output := a ^ b
	return []float64{float64(a) / 1000.0, float64(b) / 1000.0}, []float64{float64(output) / 1000.0}
}

func basicTest() {
	network := neural.MakeNetwork([]int{2, 2, 1}, 2)
	error := 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.Output(input)
		error += (answer[0] - output[0]) * (answer[0] - output[0])
	}
	fmt.Println(network)
	fmt.Println("Before:", error)
	for i := 0; i < 1000000; i++ {
		input, output := xor()
		network.Learn(input, output)
	}
	error = 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.Output(input)
		error += (answer[0] - output[0]) * (answer[0] - output[0])
	}
	fmt.Println("After:", error)

	fmt.Println(network)
}

func stringToFloats(data string, size int) []float64 {
	input := make([]float64, size)
	for i, b := range []byte(data) {
		for j := 0; j < 8; j++ {
			input[8*i+j] = float64((b & (1 << j)) >> j)
		}
	}
	return input
}

func intToFloats(data int) []float64 {
	input := make([]float64, 16)
	for j := 0; j < 16; j++ {
		input[j] = float64((data & (1 << j)) >> j)
	}
	return input
}

func testNetwork(network neural.Network, inputs []neural.Input, answers []neural.Answer) {
	start := time.Now()
	wrong := 0
	total := 0
	bitsWrong := 0
	totalBits := 0
	error := 0.0
	for i := 0; i < len(inputs); i++ {
		answer := answers[i]
		output := network.Output(inputs[i])
		isWrong := false
		for j := range answer {
			if output[j] > 0.5 {
				if answer[j] == 0 {
					isWrong = true
					bitsWrong += 1
				}
			} else {
				if answer[j] == 1 {
					isWrong = true
					bitsWrong += 1
				}
			}
			totalBits += 1
			error += (answer[j] - output[j]) * (answer[j] - output[j])
		}
		if isWrong {
			wrong += 1
		}
		total += 1
	}
	t := time.Now()
	fmt.Println("Took (s):", t.Sub(start))
	fmt.Println("Before:", error, "Wrong:", wrong, "Total:", total, "Bits wrong:", bitsWrong, "Total Bits:", totalBits)
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
		words[i] = strings.TrimSpace(string(alphabet_filter.ReplaceAll([]byte(word), []byte(""))))
		i += 1
	}
	fmt.Println(" - Found words")
	data := neural.MakeNetworkData()
	for i, word := range words {
		if len(word) <= 10 {
			data.AddData(stringToFloats(word, 100), intToFloats(i))
		}
	}
	fmt.Println(" - Converted to floats")
	trainingDataLimit := len(data.Inputs) * 9 / 10
	trainingData, testData := data.Split(trainingDataLimit)
	fastNetwork := neural.NewFastNeuralNetwork([]int{10000, 10, 16}, 100)
	testNetwork(fastNetwork, testData.Inputs, testData.Answers)
	network := neural.MakeNetwork([]int{10000, 10, 16}, 100)
	testNetwork(network, testData.Inputs, testData.Answers)
	for iteration := 0; iteration < 1; iteration++ {
		for i := 0; i < len(trainingData.Inputs); i++ {
			network.Learn(trainingData.Inputs[i], trainingData.Answers[i])
		}
	}
	testNetwork(network, testData.Inputs, testData.Answers)
}

func main() {
	initRegex()
	rand.Seed(time.Now().UnixNano())
	redditTest()
	//basicTest()
}
