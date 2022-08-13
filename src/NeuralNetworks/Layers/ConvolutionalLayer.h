#pragma once
#include <utility>
#include "Layer.h"
#include "../Neuron.h"

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

struct Filter {
    std::vector<std::shared_ptr<Weight>> weights;
    Bias bias;
};

using Field = std::vector<std::shared_ptr<Neuron>>;

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(Size inputSize, Size fieldSize, Size stride, size_t depth = 1, 
        int32_t channelCount = 1, int32_t padding = 0);
    std::vector<std::shared_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    const Size outputSize() const;
    int32_t getFieldCountPerChannel() const;
    int32_t getParamsPerKernel() const;
private:
    std::vector<std::shared_ptr<Neuron>> neurons;
    //std::vector<std::vector<Field>> channels;
    //std::vector<std::vector<Field>> fields;
    //std::vector<std::vector<std::shared_ptr<Neuron>>> fields;
    std::vector<Filter> filters;
    std::vector<Field> fields;

    const Size inputSize;
    const Size fieldSize;
    const Size stride;
    const size_t depth;
    const int32_t channelCount;
    const int32_t padding;
};
