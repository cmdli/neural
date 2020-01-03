package main

import (
	"fmt"
	"math"
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

func intToFloats(data int, numBits int) []float64 {
	input := make([]float64, numBits)
	for j := 0; j < numBits; j++ {
		input[j] = float64((data & (1 << j)) >> j)
	}
	return input
}

func numBitsForInt(num int) int {
	return int(math.Ceil(math.Log2(float64(num))))
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
		// fmt.Println(answer, output)
		total += 1
	}
	t := time.Now()
	fmt.Println("Took (s):", t.Sub(start))
	fmt.Println("Before:", error, "Wrong:", wrong, "Total:", total, "Bits wrong:", bitsWrong, "Total Bits:", totalBits)
}

func redditCommentData() (*neural.NetworkData, error) {
	comments, err := reddit.ReadComments(os.Args[1], 10000)
	if err != nil {
		return nil, err
	}
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
	data := neural.MakeNetworkData()
	for i, word := range words {
		if len(word) <= 10 {
			data.AddData(stringToFloats(word, 100), intToFloats(i, 16))
		}
	}
	return &data, nil
}

func redditTest() {
	fmt.Println(" - Starting test")
	data, err := redditCommentData()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(" - Loaded Reddit Comment Data")
	trainingData, testData := data.Split(len(data.Inputs) * 9 / 10)
	network := neural.MakeNetwork([]int{10000, 10, 16}, 100)
	testNetwork(network, testData.Inputs, testData.Answers)
	network.LearnData(trainingData, 1)
	testNetwork(network, testData.Inputs, testData.Answers)
}

func logarithmicData(num int, size int) *neural.NetworkData {
	inputs := make([]neural.Input, num)
	answers := make([]neural.Answer, num)
	for i := range inputs {
		inputs[i] = make([]float64, size)
		val := rand.Intn(size)
		inputs[i][val] = 1.0
		answers[i] = intToFloats(val, numBitsForInt(size))
	}
	return &neural.NetworkData{Inputs: inputs, Answers: answers}
}

func logarithmicTest() {
	fmt.Println(" - Starting test")
	num := 100000
	size := 1000
	data := logarithmicData(num, size)
	network := neural.MakeNetwork([]int{numBitsForInt(size)}, size)
	testNetwork(network, data.Inputs, data.Answers)
	network.LearnData(data, 1)
	testNetwork(network, data.Inputs, data.Answers)
}

func main() {
	initRegex()
	rand.Seed(time.Now().UnixNano())
	// redditTest()
	// basicTest()
	logarithmicTest()
}
