#pragma once
#include "Layer.h"

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int32_t channels, Size inputSize, Size fieldSize, Size stride, int32_t depth, int32_t padding = 0);
    std::vector<std::shared_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    int32_t getFilterCount() const;
    int32_t getNeuronsPerFilter() const;
private:
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::vector<std::vector<std::shared_ptr<Neuron>>> fields;
    const int32_t channels;
    const Size inputSize;
    const Size fieldSize;
    const Size stride; // add channel parameter and make stride a scalar?
    const int32_t padding;
    const int32_t depth;
};
