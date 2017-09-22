package main

import (
	"bufio"
	"fmt"
	"github.com/ThibaultRiviere/go-nn/pkgs/nn"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type Entry struct {
	name    string
	inputs  []float64
	outputs []float64
}

func getOutput(name string) []float64 {
	if name == "Iris-setosa" {
		return []float64{1, 0, 0}
	} else if name == "Iris-versicolor" {
		return []float64{0, 1, 0}
	} else {
		return []float64{0, 0, 1}
	}
}

func parseLine(line string) *Entry {
	arr := strings.Split(line, ",")
	name := string(arr[len(arr)-1])

	var inputs []float64
	for i := 0; i < len(arr)-1; i++ {
		val, err := strconv.ParseFloat(arr[i], 64)
		if err != nil {
			fmt.Println("err in file format: ", err)
			os.Exit(1)
		}
		// divide value by 10, this work it is cheating ???
		inputs = append(inputs, val/10)
	}
	output := getOutput(name)
	return &Entry{name, inputs, output}
}

func main() {
	file, err := os.Open("./data.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var entries []*Entry
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			entries = append(entries, parseLine(line))
		}
	}
	fmt.Println("lebngth ", len(entries))
	net := nn.New([]int{4, 5, 5, 3})
	// 0-50 50-100 100-150
	// use 45 to keep the last 5 entries unknow to the nn
	for i := 0; i < 10000; i++ {
		for y := 0; y < 3; y++ {
			x := rand.Int()%45 + (y * 45)
			entry := entries[x]
			net.FeedForward(entry.inputs)
			net.BackProb(entry.outputs)
		}
	}
	for x := 46; x < 51; x++ {
		for y := 0; y < 3; y++ {
			entry := entries[y*45+x]
			net.FeedForward(entry.inputs)
			fmt.Println("##########################################################")
			fmt.Println("result ", net.GetResults(), "expected", entry.outputs)
			fmt.Println("##########################################################")
		}
	}

}
