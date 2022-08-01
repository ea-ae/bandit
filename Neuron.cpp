#include "Neuron.h"
#include <algorithm>

Neuron::Neuron(const ActivationFunction& activationFunction, const Layer* previousLayer, const Layer* nextLayer)
    : activationFunction(activationFunction)
{
    if (previousLayer) {
        const auto connectionsIn = previousLayer->layerSize;
        const auto connectionsOut = nextLayer ? nextLayer->layerSize : 0;

        weights = std::vector<double>(previousLayer->layerSize);
        std::generate(weights.begin(), weights.end(), [&]() {
            return activationFunction.generateRandomWeight(connectionsIn, connectionsOut); // initialize weights
        });
    }
}

void Neuron::calculate(const Layer& previousLayer) {
    double sum = bias;
    for (int i = 0; i < previousLayer.layerSize; i++) {
        sum += previousLayer.neurons[i]->value * weights[i];
    }
    value = sum;
}
