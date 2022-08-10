#include "DenseLayer.h"
#include <algorithm>
#include <memory>
#include <vector>

DenseLayer::DenseLayer(int32_t neuronCount) : Layer(neuronCount) {}

void DenseLayer::connectNextLayer(const ActivationFunction& activation, const CostFunction& cost) {
    if (nextLayer == nullptr) return;

    std::generate(nextLayer->neurons.begin(), nextLayer->neurons.end(), [&]() {
        return std::make_unique<Neuron>(&neurons, activation, cost); // dense layer, pass full vector
    });
}
