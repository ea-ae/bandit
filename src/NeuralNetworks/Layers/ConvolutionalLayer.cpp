#include "ConvolutionalLayer.h"
#include <algorithm>
#include <format>

Size::Size(int32_t x, int32_t y) : x(x), y(y) {}

ConvolutionalLayer::ConvolutionalLayer(Size inputSize, Size fieldSize, Size stride, size_t depth,
    int32_t channelCount, int32_t padding)
    : inputSize(inputSize), fieldSize(fieldSize), channels(channelCount), filters(depth),
      stride(stride), depth(depth), channelCount(channelCount), padding(padding)
{
    neurons.reserve(depth * channelCount * getFieldCountPerChannel());
    // filters.reserve(depth);

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

    for (auto channel = 0; channel < channels.size(); channel++) {
        auto channelOffset = channel * inputNeuronsPerChannel;

        channels[channel] = std::vector<Field>(getFieldCountPerChannel());
        int32_t x = 0, y = 0;
        for (auto& field : channels[channel]) { // create all the field vectors
            for (int row = y; row < y + fieldSize.y; row++) {
                for (int col = x; col < x + fieldSize.x; col++) {
                    auto i = inputSize.x * row + col; // calculate coordinate pos within channel
                    auto ii = i + channelOffset; // offset by channel n
                    field.push_back(std::shared_ptr<Neuron>(prevNeurons[ii]));
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
    }
    
    for (auto& filter : filters) { // initialize the kernels/weights/neurons of each filter
        for (auto kernel = 0; kernel < channelCount; kernel++) { //  amount of kernels per filter = amount of channels
            auto& channelFields = channels[kernel]; // the fields for this kernel

            auto weights = std::vector<std::shared_ptr<Weight>>();
            for (auto i = 0; i < getParamsPerKernel() - 1; i++) { // subtract the bias param
                weights.push_back(std::make_shared<Weight>());
            }

            filter.push_back(Kernel( // create shared weights & bias for the kernel
                std::move(weights),
                Bias()
            ));

            auto& [sharedWeights, sharedBias] = filter.back(); // todo: skip the filter vector bc we dont need it w shared_ptr<Bias>

            for (auto& field : channelFields) { // create a neuron for each field in the channel/kernel for this filter
                neurons.push_back(std::make_shared<Neuron>(&field, activation, cost, &sharedWeights, &sharedBias));
            }
        }
    }

    std::cout << std::format("CL: {} neurons, {} params\n",
        depth * getFieldCountPerChannel(), depth * channelCount * getParamsPerKernel());
}

const Size ConvolutionalLayer::outputSize() const {
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    return Size(columns, rows);
}

int32_t ConvolutionalLayer::getFieldCountPerChannel() const {
    int32_t rows = (2 * padding + inputSize.y - fieldSize.y) / stride.y + 1;
    int32_t columns = (2 * padding + inputSize.x - fieldSize.x) / stride.x + 1;
    return rows * columns;
}

int32_t ConvolutionalLayer::getParamsPerKernel() const {
    return fieldSize.x * fieldSize.y + 1;
}
