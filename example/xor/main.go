package main

import "fmt"
import "math/rand"
import "github.com/ThibaultRiviere/go-nn"

func testNN(n *nn.Network, input []float64, expected float64) {
	fmt.Println("input : ", input)
	n.FeedForward(input)
	fmt.Println("result :", n.GetResults())
	fmt.Println("expected :", expected, "\n")
	n.BackProb([]float64{expected})
}

type test struct {
	input  []float64
	output float64
}

func main() {
	possible := make([]test, 4)
	possible[0] = test{[]float64{0.0, 0.0}, 0.0}
	possible[1] = test{[]float64{0.0, 1.0}, 1.0}
	possible[2] = test{[]float64{1.0, 0.0}, 1.0}
	possible[3] = test{[]float64{1.0, 1.0}, 0.0}
	net := nn.New([]int{2, 4, 1})
	for i := 0; i < 1000; i++ {
		index := rand.Intn(len(possible))
		pos := possible[index]
		testNN(net, pos.input, pos.output)
	}

	//	net := New([]int{2, 4, 4, 1})
	//	for i := 0; i < 100000; i++ {
	//		index := float64(rand.Intn(10000))
	//		index2 := float64(rand.Intn(10000))
	//		testNN(net, []float64{index, index2}, index*index2)
	//	}

}
