#pragma once
#include <utility>
#include "Layer.h"
#include "../Neuron.h"



struct Filter {
    std::vector<std::shared_ptr<Weight>> weights;
    Bias bias;
};

using Field = std::vector<std::shared_ptr<Neuron>>;

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(Size3 inputSize, Size3 fieldSize, Size stride = Size(1, 1), int32_t padding = 0);
    std::vector<std::shared_ptr<Neuron>>& getNeurons();
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    const Size3 outputSize() const;
    int32_t getFieldCountPerChannel() const;
    int32_t getParamsPerKernel() const;
private:
    std::vector<std::shared_ptr<Neuron>> neurons;
    //std::vector<std::vector<Field>> channels;
    //std::vector<std::vector<Field>> fields;
    //std::vector<std::vector<std::shared_ptr<Neuron>>> fields;
    std::vector<Filter> filters;
    std::vector<Field> fields;

    const Size3 inputSize;
    const Size3 fieldSize;
    const Size stride;
    const int32_t padding;
    const int32_t channelCount;
    const size_t depth;
};
