#pragma once
#include <vector>
#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(int32_t neuronCount);
    std::vector<std::unique_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    const Size3 outputSize() const;
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
};
