#pragma once
#include "Relu.h"

class LeakyRelu : public Relu {
private:
    double leakRate;
public:
    LeakyRelu();
    LeakyRelu(double leakRate);
    double map(double input) const;
    double getPreValueDerivative(double input) const;
};
