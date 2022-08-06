#pragma once
#include "NeuralNetwork.h"

class ClassificationNeuralNetwork : public NeuralNetwork {
public:
    using NeuralNetwork::NeuralNetwork;
    int32_t getHighestOutputNode();
    float calculateCost(int32_t label);
private:
    float getExpectedValue(int32_t label, int32_t neuronIndex);
};
