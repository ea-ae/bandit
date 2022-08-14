#include "Layer.h"

#include <algorithm>
#include <numeric>

Size::Size(int32_t x, int32_t y) : x(x), y(y) {}

Size3::Size3(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : getNeurons()) {
            neuron->calculate();
        }
    }
    if (nextLayer) nextLayer->calculateNodeValues();
}

void Layer::update(float learningRate) {
    if (!previousLayer) return;  // don't update input nodes

    for (auto& neuron : getNeurons()) neuron->update(learningRate);

    previousLayer->update(learningRate);  // move onto next layer
}

size_t Layer::getWeightCount() {
    return std::accumulate(getNeurons().begin(), getNeurons().end(), static_cast<size_t>(0),
                           [](size_t sum, auto& neuron) {
                               return sum + neuron->getWeightCount();
                           });
}
