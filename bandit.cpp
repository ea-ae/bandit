#include <iostream>
#include "NeuralNetwork.h"
#include "ActivationFunctions/Relu.h"

int main()
{
    std::cout << "Hello World!\n";

    const auto activationFunction = Relu();
    const auto inputNodes = 3;
    const auto outputNodes = 3;

    auto net = NeuralNetwork(activationFunction, inputNodes, outputNodes, 0, 0);
}
