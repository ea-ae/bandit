#include "QuadraticCost.h"

#include <cmath>

// float QuadraticCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const { // todo int->float
//     float totalCost = 0.0;
//     for (int i = 0; i < outputLayer.layerSize; i++) {
//         auto a = outputLayer.neurons[i]->value;
//         auto y = expected[i];
//         float cost = std::powf(a - y, 2) / 2;
//         totalCost += cost;
//     }
//     return totalCost;
// }
