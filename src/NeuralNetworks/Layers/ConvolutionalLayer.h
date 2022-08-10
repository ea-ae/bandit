#pragma once
#include "Layer.h"

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

class ConvolutionalLayer : public Layer { // ps: add zero padding
public:
    ConvolutionalLayer(int32_t neuronCount, int32_t channels, Size inputSize, Size fieldSize, Size stride, int32_t depth);
    void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost);
    int32_t getFilterCount() const;
private:
    const int32_t channels;
    const Size inputSize;
    const Size fieldSize;
    const Size stride; // add channel parameter and make stride a scalar?
    const int32_t depth;
};
