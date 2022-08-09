#include "Neuron.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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
    values = values.setConstant(bias);
    for (int i = 0; i < previousLayer.layerSize; i++) {
        auto& inputs = previousLayer.neurons[i]->values;
        values += inputs * weights[i].weight; // scalar multiplication
    }
    preTransformedValues = values;
    values = values.unaryExpr([this](float preValue) { return activationFunction.map(preValue); });
}

void Neuron::addActivationGradients(const BatchArray& gradients) {
    activationGradients += gradients;
}

void Neuron::backpropagate(Layer& previousLayer, BatchArray* expectedValues) {
    if (expectedValues != nullptr) { // are we in an output node?
        // activationGradients = costFunction.getActivationDerivatives(values, *expectedValues); // dC/da = 2(a - y)
        activationGradients = values - *expectedValues; // dC/da = 2(a - y)
    } // else, we apply precalculated dC/da1 * da1/dz1 * dz1/da2 chain rule result to our derivatives

    auto activationsDerivedByPreValues = values.unaryExpr([this](float value) {
        return activationFunction.getPreValueDerivative(value); // da/dz
    });
    auto costDerivedByPreValues = activationsDerivedByPreValues * activationGradients; // C->a1->z1, C->a1->z1->a2->z2, etc

    biasGradient = costDerivedByPreValues.sum(); // dC/dz = dC/db

    for (int i = 0; i < previousLayer.layerSize; i++) { // calculate weights and biases for each neuron in previous layer
        auto& preValuesDerivedByWeight = previousLayer.neurons[i]->values;
        auto weightGradient = (preValuesDerivedByWeight * costDerivedByPreValues).sum();
        weights[i].gradient += weightGradient;

        if (previousLayer.previousLayer) { // first check to make sure the next layer isn't the input layer
            // find the new activation derivative for the next layer of backpropagation
            auto preValuesDerivedByActivation = weights[i].weight;
            auto costDerivedByActivations = preValuesDerivedByActivation * costDerivedByPreValues;
            previousLayer.neurons[i]->addActivationGradients(costDerivedByActivations); // add the gradient
        }
    }

    activationGradients = activationGradients.setZero();
}

void Neuron::update(int32_t batchSize, float learningRate) {
    float biasDelta = (biasGradient / batchSize) * learningRate;
    bias -= biasDelta;

    for (auto& weight : weights) {
        auto regularizationTerm = costFunction.getRegularizationDerivative(weight.weight, batchSize);
        auto weightGradient = (weight.gradient / batchSize + regularizationTerm) * learningRate;

        momentum = costFunction.getMomentum(momentum, weightGradient); // mu * v - eta * delC
        weight.weight -= momentum;
        weight.gradient = 0;
    }
}

size_t Neuron::getWeightCount() {
    return weights.size();
}
