#pragma once
#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(int32_t neuronCount);
    std::vector<std::unique_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
};
