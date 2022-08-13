#include "ConvolutionalLayer.h"
#include <algorithm>
#include <format>

ConvolutionalLayer::ConvolutionalLayer(Size3 inputSize, Size3 fieldSize, Size stride, int32_t padding)
    : inputSize(inputSize), fieldSize(fieldSize), filters(fieldSize.z),
      stride(stride), depth(fieldSize.z), channelCount(inputSize.z), padding(padding)
{
    neurons.reserve(depth * getFieldCountPerChannel());

    auto pad = 2 * padding;
    if (2 * padding + fieldSize.x > inputSize.x || 2 * padding + fieldSize.y > inputSize.y) {
        throw std::logic_error("Receptive field size is larger than the x/y size arguments");
    } else if (stride.x > fieldSize.x || stride.y > fieldSize.y) {
        throw std::logic_error("Stride is larger than receptive field size and therefore skips inputs");
    } else if ((pad + inputSize.x - fieldSize.x) % stride.x != 0 || (pad + inputSize.y - fieldSize.y) % stride.y != 0) {
        throw std::logic_error("Stride results in asymmetric outputs due to skipped inputs");
    }
}

std::vector<std::shared_ptr<Neuron>>& ConvolutionalLayer::getNeurons() {
    return neurons;
}

void ConvolutionalLayer::connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) {
    auto& prevNeurons = previousLayer->getNeurons();

    auto inputNeuronsPerChannel = inputSize.x * inputSize.y;

    if (prevNeurons.size() != static_cast<size_t>(inputNeuronsPerChannel) * channelCount) {
        throw std::logic_error("Incorrect input size (area mismatch)");
    } else if (prevNeurons.size() % channelCount != 0) {
        throw std::logic_error("Invalid channel count (doesn't divide by neuron count)");
    }

    fields = std::vector<Field>(getFieldCountPerChannel()); // e.g. vector of 5x5x3 fields

    for (auto channel = 0; channel < channelCount; channel++) {
        auto channelOffset = channel * inputNeuronsPerChannel;
        int32_t x = 0, y = 0;
        for (auto field = 0; field < getFieldCountPerChannel(); field++) {
            for (int32_t row = y; row < y + fieldSize.y; row++) {
                for (int32_t col = x; col < x + fieldSize.x; col++) {
                    auto i = inputSize.x * row + col + channelOffset; // calculate coordinate pos within channel + offset by channel n
                    fields[field].push_back(std::shared_ptr<Neuron>(prevNeurons[i])); // add input neuron for field
                }
            }

            // stride onto next field
            x += stride.x;
            if (x + fieldSize.x > inputSize.x) {
                x = 0;
                y += stride.y;
            }
        }
    }
    
    for (auto& filter : filters) { // filter depth is analogous to the amount of input channels in the next convolutional layer
        for (auto kernel = 0; kernel < channelCount; kernel++) { //  amount of kernels per filter = amount of channels
            for (auto i = 0; i < fieldSize.x * fieldSize.y; i++) {
                filter.weights.push_back(std::make_shared<Weight>()); // shared weights for the filter, e.g. 5x5x3 for W=H=5 and D=3
            }
        }

        for (auto& field : fields) { // each field has inputs from every channel
            neurons.push_back(std::make_shared<Neuron>(&field, activation, cost, &filter.weights, &filter.bias));
        }
    }

    std::cout << std::format("CL | {} neurons, {} params\t {:02}x{:02}x{:02} & f{}x{} s{}x{} -> {:02}x{:02}x{:02}\n",
        depth * getFieldCountPerChannel(), depth * channelCount * getParamsPerKernel(),
        inputSize.x, inputSize.y, inputSize.z, fieldSize.x, fieldSize.y, stride.x, stride.y, 
        outputSize().x, outputSize().y, outputSize().z);
}

const Size3 ConvolutionalLayer::outputSize() const {
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    return Size3(columns, rows, fieldSize.z);
}

int32_t ConvolutionalLayer::getFieldCountPerChannel() const {
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    return rows * columns;
}

int32_t ConvolutionalLayer::getParamsPerKernel() const {
    return fieldSize.x * fieldSize.y + 1;
}
