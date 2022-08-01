#pragma once
#include "NeuralNetwork.h"

class SupervisedNeuralNetwork : public NeuralNetwork {
public:
    using NeuralNetwork::NeuralNetwork;
    int32_t getHighestOutputNode();
    double calculateCost(int32_t label);
    void backpropagate(int32_t label);
};
