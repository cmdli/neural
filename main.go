package main

import (
	"fmt"
	"math/rand"
	"time"
	"os"
	"./reddit"
	"./neural"
	"strings"
)

func xor() ([]float64, []float64) {
	a := rand.Intn(1000)
	b := rand.Intn(1000)
	output := a ^ b
	return []float64{float64(a)/1000.0, float64(b)/1000.0}, []float64{float64(output)/1000.0}
}


func basicTest() {
	network := neural.MakeNetwork([]int{2,2,1},2)
	error := 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.Output(input)
		error += (answer[0]-output[0])*(answer[0]-output[0])
	}
	fmt.Println(network)
	fmt.Println("Before:",error)
	for i := 0; i < 1000000; i++ {
		input, output := xor()
		network.Learn(input,output)
	}
	error = 0.0
	for i := 0; i < 100; i++ {
		input, answer := xor()
		output := network.Output(input)
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
	network := neural.MakeNetwork([]int{10,32}, 100)
	trainingDataLimit := len(inputs)*9/10
	for i := 0; i < trainingDataLimit; i++ {
		network.Learn(inputs[i], outputs[i])
	}
	fmt.Println("Input:",inputs[trainingDataLimit])
	fmt.Println("Output:",network.Output(inputs[trainingDataLimit]))
	fmt.Println("Expected:",outputs[trainingDataLimit])
	fmt.Println(network)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	redditTest()
	//basicTest()
}