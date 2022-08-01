#include "Layer.h"
#include <algorithm>

Layer::Layer(const ActivationFunction& activationFunction, int32_t layerSize, Layer* previousLayer, Layer* nextLayer)
    : neurons(layerSize), layerSize(layerSize), previousLayer(previousLayer), nextLayer(nextLayer)
{
    neurons = std::vector<std::unique_ptr<Neuron>>(layerSize);
    std::generate(neurons.begin(), neurons.end(), [&]() {
        return std::make_unique<Neuron>(activationFunction, previousLayer, nextLayer);
    });
}

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : neurons) {
            neuron->calculate(*previousLayer);
        }
    }
    if (nextLayer) nextLayer->calculateNodeValues();
}
