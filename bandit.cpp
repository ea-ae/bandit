#include <iostream>
#include "NeuralNetwork.h"
#include "ActivationFunctions/Relu.h"

int main()
{
    std::cout << "Hello World!\n";

    const auto activationFunc = Relu();
    const auto net = NeuralNetwork(activationFunc, 3, 3, 0, 0);
}
