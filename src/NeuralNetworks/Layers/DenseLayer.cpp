#include "DenseLayer.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <format>

DenseLayer::DenseLayer(int32_t neuronCount) : neurons(neuronCount) {}

std::vector<std::shared_ptr<Neuron>>& DenseLayer::getNeurons() {
    return neurons;
}

void DenseLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {
    if (previousLayer == nullptr) return;

    std::generate(neurons.begin(), neurons.end(), [&]() { // dense layer, pass full vector
        return std::make_shared<Neuron>(&previousLayer->getNeurons(), activation, cost);
    });

    std::cout << std::format("DL: {} neurons, {} params\n", neurons.size(), getWeightCount() + neurons.size());
}
