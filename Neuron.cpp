#include "Neuron.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Neuron::Neuron(const ActivationFunction& activationFunction, const Layer* previousLayer, const Layer* nextLayer)
    : activationFunction(activationFunction)
{
    if (previousLayer) {
        const auto connectionsIn = previousLayer->layerSize;
        const auto connectionsOut = nextLayer ? nextLayer->layerSize : 0;

        weights = std::vector<Weight>(previousLayer->layerSize);
        std::generate(weights.begin(), weights.end(), [&]() { // initialize weights
            Weight weight;
            weight.weight = activationFunction.generateRandomWeight(connectionsIn, connectionsOut);
            return weight;
        });
    }
}

void Neuron::calculate(const Layer& previousLayer) {
    double sum = bias;
    for (int i = 0; i < previousLayer.layerSize; i++) {
        sum += previousLayer.neurons[i]->value * weights[i].weight;
    }
    preTransformedValue = sum;
    value = activationFunction.map(preTransformedValue);
}

void Neuron::backpropagate(Layer& previousLayer, std::optional<double> expectedValue, double activationDerivative) {
    if (expectedValue.has_value()) { // are we in an output node?
        activationDerivative = 2 * (value - expectedValue.value()); // dC/da = 2(a - y)
    } // else, we apply precalculated dC/da1 * da1/dz1 * dz1/da2 chain rule result to our derivatives

    auto valueDerivedByPreValue = activationFunction.derivative(value); // da/dz
    auto costDerivedByPreValue = valueDerivedByPreValue * activationDerivative; // e.g. C->a1->z1, C->a1->z1->a2->z2, etc

    auto preValueDerivedByBias = 1;
    biasGradient += preValueDerivedByBias * costDerivedByPreValue; // for non-output nodes, we sum all the output influences
    if (std::isnan(biasGradient)) {
        int x = 666;
    }

    for (int i = 0; i < previousLayer.layerSize; i++) { // calculate weights and biases for each neuron in previous layer
        auto preValueDerivedByWeight = previousLayer.neurons[i]->value;
        auto weightGradient = preValueDerivedByWeight * costDerivedByPreValue;
        weights[i].gradient += weightGradient;

        if (previousLayer.previousLayer) { // first check to make sure the next layer isn't the input one
            // find the new activation derivative for the next layer of backpropagation
            auto preValueDerivedByActivation = weights[i].weight;
            auto costDerivedByActivation = preValueDerivedByActivation * costDerivedByPreValue;

            if (std::isinf(costDerivedByActivation)) {
                int y = 666;
            }

            previousLayer.neurons[i]->backpropagate(*previousLayer.previousLayer, {}, costDerivedByActivation);
        }
    }
}

void Neuron::update(double batchSize, double learningRate) {
    bias -= (biasGradient / batchSize) * learningRate;
    if (std::isnan(bias)) {
        int x = 666;
    }
    biasGradient = 0;

    for (auto& weight : weights) {
        weight.weight -= (weight.gradient / batchSize) * learningRate;
        weight.gradient = 0;
    }
}
