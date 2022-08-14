#pragma once
#pragma warning(push, 0)
#include <eigen/Eigen/Dense>
#pragma warning( pop )

const auto BATCH_SIZE = 16;

using BatchArray = Eigen::Array<float, 1, BATCH_SIZE>;
using BatchLabelArray = Eigen::Array<int16_t, 1, BATCH_SIZE>;

void main();
