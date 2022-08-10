#include "ConvolutionalLayer.h"

Size::Size(int32_t x, int32_t y) : x(x), y(y) {}

ConvolutionalLayer::ConvolutionalLayer(int32_t channels, Size inputSize, Size fieldSize, 
    Size stride, int32_t padding, int32_t depth)
    : channels(channels), inputSize(inputSize), fieldSize(fieldSize), stride(stride), padding(padding), depth(depth)
{
    neurons = std::vector<std::unique_ptr<Neuron>>(getFilterCount() * getNeuronsPerFilter());

    if (2 * padding + fieldSize.x > inputSize.x || 2 * padding + fieldSize.y > inputSize.y) {
        throw std::logic_error("Receptive field size is larger than the x/y size arguments");
    }
    if (stride.x > fieldSize.x || stride.y > fieldSize.y) {
        throw std::logic_error("Stride is larger than receptive field size and skips inputs");
    }
    auto pad = 2 * padding;
    if ((pad + inputSize.x - fieldSize.x) % stride.x != 0 || (pad + inputSize.y - fieldSize.y) % stride.y != 0) {
        throw std::logic_error("Stride results in asymmetric outputs due to skipped inputs");
    }
}

std::vector<std::unique_ptr<Neuron>>& ConvolutionalLayer::getNeurons() {
    return neurons;
}

void ConvolutionalLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {

}

int32_t ConvolutionalLayer::getFilterCount() const {
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    return columns * rows * depth;
}

int32_t ConvolutionalLayer::getNeuronsPerFilter() const {
    return channels * fieldSize.x * fieldSize.y;
}
