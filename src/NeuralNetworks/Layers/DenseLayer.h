#pragma once
#include "Layer.h"

class DenseLayer : public Layer {
public:
    DenseLayer(int32_t neuronCount);
    void connectNextLayer(const ActivationFunction& activation, const CostFunction& cost);
};
