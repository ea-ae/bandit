#include "DenseLayer.h"
#include <algorithm>
#include <memory>
#include <vector>

DenseLayer::DenseLayer(int32_t neuronCount) : Layer(neuronCount) {}

void DenseLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {
    if (previousLayer == nullptr) return;

    std::generate(neurons.begin(), neurons.end(), [&]() {
        return std::make_unique<Neuron>(&previousLayer->neurons, activation, cost); // dense layer, pass full vector
    });
}
