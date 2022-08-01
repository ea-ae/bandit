#include "Layer.h"
#include <algorithm>

Layer::Layer(const ActivationFunction& activationFunction, int32_t layerSize, Layer& previousLayer, Layer& nextLayer)
    : neurons(layerSize), layerSize(layerSize), previousLayer(previousLayer), nextLayer(nextLayer)
{
    // neurons = std::vector<std::unique_ptr<Neuron>>(layerSize);
    //std::generate(neurons.begin(), neurons.end(), []() { return std::make_unique<Neuron>(); });
}
