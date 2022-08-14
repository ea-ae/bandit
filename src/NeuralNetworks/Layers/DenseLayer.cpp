#include "DenseLayer.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <format>
#include "../DenseNeuron.h"

DenseLayer::DenseLayer(int32_t neuronCount) : neurons(neuronCount) {}

std::vector<std::unique_ptr<Neuron>>& DenseLayer::getNeurons() {
    return neurons;
}

const Size3 DenseLayer::outputSize() const {
    return Size3(neurons.size(), 1, 1);
}

void DenseLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {
    if (previousLayer == nullptr) return;

    std::generate(neurons.begin(), neurons.end(), [&]() { // dense layer, pass full vector
        return std::make_unique<DenseNeuron>(&previousLayer->getNeurons(), activation, cost);
    });

    std::cout << std::format("DL | {} neurons, {} params\n", neurons.size(), getWeightCount() + neurons.size());
}
