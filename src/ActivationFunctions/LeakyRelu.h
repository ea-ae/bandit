#pragma once
#include "Relu.h"

class LeakyRelu : public Relu {
   private:
    float leakRate;

   public:
    LeakyRelu();
    LeakyRelu(float leakRate);
    float map(float input) const;
    float getPreValueDerivative(float input) const;
};
