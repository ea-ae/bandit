#include "ConvolutionalLayer.h"
#include <algorithm>

Size::Size(int32_t x, int32_t y) : x(x), y(y) {}

ConvolutionalLayer::ConvolutionalLayer(int32_t channels, Size inputSize, Size fieldSize, 
    Size stride, int32_t depth, int32_t padding)
    : channels(channels), inputSize(inputSize), fieldSize(fieldSize), stride(stride), depth(depth), padding(padding)
{
    //neurons = std::vector<std::unique_ptr<Neuron>>(getFilterCount() * getNeuronsPerFilter());
    neurons.reserve(getFilterCount());
    fields = std::vector<std::vector<std::shared_ptr<Neuron>>>(getFilterCount() / depth);

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

std::vector<std::shared_ptr<Neuron>>& ConvolutionalLayer::getNeurons() {
    return neurons;
}

void ConvolutionalLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {
    auto& prevNeurons = previousLayer->getNeurons();
    int32_t x = 0, y = 0;

    for (auto& field : fields) { // create all the field vectors
        for (int row = y; row < y + fieldSize.y; row++) {
            for (int col = x; col < x + fieldSize.x; col++) {
                auto i = inputSize.x * row + col;
                field.push_back(std::shared_ptr<Neuron>(prevNeurons[i]));
                // std::cout << i << " ";
            }
        }
        // std::cout << "\n";

        // stride onto next field
        x += stride.x;
        if (x + fieldSize.x > inputSize.x) {
            x = 0;
            y += stride.y;
        }
    }

    // create the filters/kernels on top of the fields
    for (int i = 0; i < depth; i++) { // make 'depth' amount of filters per field
        for (auto& field : fields) {
            neurons.push_back(std::make_shared<Neuron>(&field, activation, cost));
        }
    }

    std::cout << "CL: " << getFilterCount() << " neurons/filters, " << (getNeuronsPerFilter() + getFilterCount()) << " params\n";
}

int32_t ConvolutionalLayer::getFilterCount() const {
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    return rows * columns * depth;
}

int32_t ConvolutionalLayer::getNeuronsPerFilter() const {
    return channels * fieldSize.x * fieldSize.y;
}
