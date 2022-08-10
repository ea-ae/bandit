#include "Layer.h"
#include <algorithm>
#include <numeric>

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : getNeurons()) {
            neuron->calculate();
        }
    }
    if (nextLayer) nextLayer->calculateNodeValues();
}

void Layer::update(float learningRate) {
    if (!previousLayer) return; // don't update input nodes

    for (auto& neuron : getNeurons()) neuron->update(learningRate);

    previousLayer->update(learningRate); // move onto next layer
}

size_t Layer::getWeightCount() {
    return std::accumulate(getNeurons().begin(), getNeurons().end(), (size_t)0, [](size_t sum, auto& neuron) {
        return sum + neuron->getWeightCount();
    });
}
