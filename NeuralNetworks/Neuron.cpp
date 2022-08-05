#include "Neuron.h"
#include <algorithm>
#include <cmath>

Neuron::Neuron(const ActivationFunction& activationFunction, const CostFunction& costFunction,
    const Layer* previousLayer, const Layer* nextLayer)
    : activationFunction(activationFunction), costFunction(costFunction)
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

void Neuron::addActivationGradient(double gradient) {
    activationGradient += gradient;
}

void Neuron::backpropagate(Layer& previousLayer, std::optional<double> expectedValue) {
    if (expectedValue.has_value()) { // are we in an output node?
        activationGradient = costFunction.getActivationDerivative(value, expectedValue.value()); // dC/da = 2(a - y)
    } // else, we apply precalculated dC/da1 * da1/dz1 * dz1/da2 chain rule result to our derivatives

    auto activationDerivedByPreValue = activationFunction.getPreValueDerivative(value); // da/dz
    auto costDerivedByPreValue = activationDerivedByPreValue * activationGradient; // C->a1->z1, C->a1->z1->a2->z2, etc

    double preValueDerivedByBias = 1.0;
    biasGradient += preValueDerivedByBias * costDerivedByPreValue;

    for (int i = 0; i < previousLayer.layerSize; i++) { // calculate weights and biases for each neuron in previous layer
        auto preValueDerivedByWeight = previousLayer.neurons[i]->value;
        auto weightGradient = preValueDerivedByWeight * costDerivedByPreValue;
        weights[i].gradient += weightGradient + costFunction.getRegularizationDerivative(weights[i].weight);

        if (previousLayer.previousLayer) { // first check to make sure the next layer isn't the input layer
            // find the new activation derivative for the next layer of backpropagation
            auto preValueDerivedByActivation = weights[i].weight;
            auto costDerivedByActivation = preValueDerivedByActivation * costDerivedByPreValue;

            previousLayer.neurons[i]->addActivationGradient(costDerivedByActivation); // add the gradient
        }
    }

    activationGradient = 0; // reset the sum for the next cycle
}

void Neuron::update(double batchSize, double learningRate) {
    double biasDelta = (biasGradient / batchSize) * learningRate;
    bias -= biasDelta;
    biasGradient = 0;

    for (auto& weight : weights) {
        weight.weight -= (weight.gradient / batchSize) * learningRate;
        weight.gradient = 0;
    }
}

size_t Neuron::getWeightCount() {
    return weights.size();
}
