#include <iostream>
#include <eigen/Eigen/Dense>
#include "mnist.h"
//#include "imagenet.h"

const int32_t BATCH_SIZE = 16;

int main()
{
    std::cout << "Initializing\n";
    using ActivationVector = Eigen::Matrix<float, 1, BATCH_SIZE>;
    /*auto a = ActivationVector(ActivationVector::Zero());
    auto b = ActivationVector(ActivationVector::Constant(5));
    auto c = ActivationVector(ActivationVector::Constant(1.5));*/
    
    using A = Eigen::Array<float, 1, BATCH_SIZE>;
    auto a = A(A::Zero());
    auto b = A(A::Constant(5));
    auto c = A(A::Constant(3));
    std::cout << "b: " << b << "\n";
    std::cout << "c: " << c << "\n";
    std::cout << "2 * b: " << 2 * c << "\n";
    std::cout << "b * c: " << b * c << "\n";
    std::cout << "b / c: " << b / c << "\n";

    // mnist();
    // imagenet();
}
