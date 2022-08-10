#include "cifar100.h"
#include <format.>
#include "DataLoaders/CifarDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "NeuralNetworks/Layers/DenseLayer.h"
#include "NeuralNetworks/Layers/ConvolutionalLayer.h"
#include "Trainers/ClassificationTrainer.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"
#include "NeuralNetworks/Neuron.h"

void cifar100() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.02f; // default: 0.1-0.2
    const auto MOMENTUM_COEFFICIENT_MU = 0.9f; // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto RELU_LEAK = 0.01f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    //net.addLayer(new ConvolutionalLayer(1, Size(28, 28), ...);
    //net.addLayer(new ConvolutionalLayer(Size(28, 28), Size(6, 6), Size(2, 2), 20));
    net.addLayer(new DenseLayer(300));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = CifarDataLoader("./train.bin", 50000);
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = CifarDataLoader("./test.bin", 10000);
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);
}
