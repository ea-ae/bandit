#include "Neuron.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Neuron::Neuron(std::vector<std::shared_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost)
    : inputNeurons(inputNeurons), activationFunction(activation), costFunction(cost)
{
    if (inputNeurons != nullptr) {
        selfWeights = std::vector<Weight>(inputNeurons->size());
        std::generate(selfWeights.begin(), selfWeights.end(), [&]() { // initialize weights
            Weight weight;
            weight.weight = activationFunction.generateRandomWeight(inputNeurons->size());
            return weight;
        });
        weights = &selfWeights;
        bias = &selfBias;
    }
}

Neuron::Neuron(std::vector<std::shared_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost,
    std::vector<Weight>* sharedWeights, Bias* sharedBias)
    : inputNeurons(inputNeurons), activationFunction(activation), costFunction(cost), weights(sharedWeights), bias(sharedBias) {}

void Neuron::calculate() {
    values = values.setConstant(bias->bias);
    for (int i = 0; i < inputNeurons->size(); i++) { // a*w dot product
        values += (*inputNeurons)[i]->values * (*weights)[i].weight; // scalar multiplication
    }
    preTransformedValues = values;
    values = values.unaryExpr([this](float preValue) { return activationFunction.map(preValue); });
}

void Neuron::addActivationGradients(const BatchArray& gradients) {
    activationGradients += gradients;
}

void Neuron::backpropagate(bool backpropagateGradients, BatchArray* expectedValues) {
    if (expectedValues != nullptr) { // are we in an output node?
        activationGradients = values - *expectedValues; // dC/da = 2(a - y)
    } // else, we apply precalculated dC/da1 * da1/dz1 * dz1/da2 chain rule result to our derivatives

    auto activationsDerivedByPreValues = values.unaryExpr([this](float value) {
        return activationFunction.getPreValueDerivative(value); // da/dz
    });
    auto costDerivedByPreValues = activationsDerivedByPreValues * activationGradients; // C->a1->z1, C->a1->z1->a2->z2, etc

    bias->gradient += costDerivedByPreValues.sum(); // dC/dz = dC/db
    bias->done = false;

    for (int i = 0; i < inputNeurons->size(); i++) { // calculate weights and biases for each neuron in previous layer
        auto& preValuesDerivedByWeight = (*inputNeurons)[i]->values;
        auto weightGradient = (preValuesDerivedByWeight * costDerivedByPreValues).sum();
        (*weights)[i].gradient += weightGradient;

        if (backpropagateGradients) { // first check to make sure the next layer isn't the input layer
            // find the new activation derivative for the next layer of backpropagation
            auto preValuesDerivedByActivation = (*weights)[i].weight;
            auto costDerivedByActivations = preValuesDerivedByActivation * costDerivedByPreValues;
            (*inputNeurons)[i]->addActivationGradients(costDerivedByActivations); // add the gradient
        }
    }

    activationGradients = activationGradients.setZero();
}

void Neuron::update(float learningRate) {
    if (bias->done) return;

    float biasGradient = (bias->gradient / BATCH_SIZE) * learningRate;
    bias->bias -= costFunction.getMomentum(bias->momentum, biasGradient);
    bias->gradient = 0;
    bias->done = true; // prevents multiple updates on shared weights

    for (auto& weight : *weights) {
        auto regularizationTerm = costFunction.getRegularizationDerivative(weight.weight);
        auto weightGradient = (weight.gradient / BATCH_SIZE + regularizationTerm) * learningRate;

        weight.momentum = costFunction.getMomentum(weight.momentum, weightGradient); // mu * v - eta * delC
        weight.weight -= weight.momentum;
        weight.gradient = 0;
    }
}

size_t Neuron::getWeightCount() {
    return weights->size();
}
