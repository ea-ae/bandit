#pragma once
#include "Layer.h"

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int32_t channels, Size inputSize, Size fieldSize, Size stride, int32_t padding, int32_t depth);
    std::vector<std::unique_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    int32_t getFilterCount() const;
    int32_t getNeuronsPerFilter() const;
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
    const int32_t channels;
    const Size inputSize;
    const Size fieldSize;
    const Size stride; // add channel parameter and make stride a scalar?
    const int32_t padding;
    const int32_t depth;
};
