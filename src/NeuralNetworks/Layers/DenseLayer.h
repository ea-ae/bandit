#pragma once
#include <vector>
#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(int32_t neuronCount);
    std::vector<std::shared_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
private:
    std::vector<std::shared_ptr<Neuron>> neurons;
};
