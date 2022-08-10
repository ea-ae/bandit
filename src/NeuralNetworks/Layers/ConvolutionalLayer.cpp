#include "ConvolutionalLayer.h"

Size::Size(int32_t x, int32_t y) : x(x), y(y) {}

ConvolutionalLayer::ConvolutionalLayer(int32_t neuronCount, 
    int32_t channels, Size inputSize, Size fieldSize, Size stride, int32_t depth)
    : Layer(neuronCount), channels(channels), inputSize(inputSize), fieldSize(fieldSize), stride(stride), depth(depth)
{
    if (inputSize.x * inputSize.y != neuronCount) {
        throw std::logic_error("The x/y size arguments do not match the neuron count");
    }
    if (fieldSize.x > inputSize.x || fieldSize.y > inputSize.y) {
        throw std::logic_error("Receptive field size is larger than the x/y size arguments");
    }
    if (stride.x > fieldSize.x || stride.y > fieldSize.y) {
        throw std::logic_error("Stride is larger than receptive field size and skips inputs");
    }
    if ((inputSize.x - fieldSize.x) % stride.x != 0 || (inputSize.y - fieldSize.y) % stride.y != 0) {
        throw std::logic_error("Stride results in asymmetric outputs due to skipped inputs");
    }
}

void ConvolutionalLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {

}

int32_t ConvolutionalLayer::getFilterCount() const {
    int32_t columns = (inputSize.x - fieldSize.x) / stride.x + 1;
    int32_t rows = (inputSize.y - fieldSize.y) / stride.y + 1;
    return columns * rows;
}
