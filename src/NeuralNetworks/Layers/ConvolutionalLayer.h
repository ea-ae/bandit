#pragma once
#include <utility>
#include "Layer.h"
#include "../Neuron.h"

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(Size inputSize, Size fieldSize, Size stride, int32_t depth = 1, 
        int32_t dimensions = 1, int32_t channels = 1, int32_t padding = 0);
    std::vector<std::shared_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    int32_t getFieldCountPerDepth() const;
    int32_t getParamsPerFilter() const;
private:
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::vector<std::vector<std::shared_ptr<Neuron>>> fields;
    std::vector<std::pair<std::vector<Weight>, Bias>> filters;
    const Size inputSize;
    const Size fieldSize;
    const Size stride;
    const int32_t depth;
    const int32_t dimensions;
    const int32_t channels;
    const int32_t padding;
};
