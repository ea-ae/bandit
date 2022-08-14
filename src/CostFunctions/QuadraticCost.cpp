#include "QuadraticCost.h"

#include <cmath>

// TODO INT TO FLOAT HERE
// float QuadraticCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const {
//     float totalCost = 0.0;
//     for (int i = 0; i < outputLayer.layerSize; i++) {
//         auto a = outputLayer.neurons[i]->value;
//         auto y = expected[i];
//         float cost = std::powf(a - y, 2) / 2;
//         totalCost += cost;
//     }
//     return totalCost;
// }
