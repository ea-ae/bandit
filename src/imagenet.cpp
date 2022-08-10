#include "imagenet.h"
#include <iostream>
#include <format>
#include "DataLoaders/ImageNetDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "NeuralNetworks/Layers/DenseLayer.h"
#include "Trainers/ClassificationTrainer.h"
#include "CostFunctions/QuadraticCost.h"
#include "ActivationFunctions/LeakyRelu.h"

void imagenet() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.1f; // default: 0.1
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto MOMENTUM_COEFFICIENT_MU = 0.1f; // no momentum: 0
    const auto RELU_LEAK = 0.1f; // no leak: 0

    const auto INPUT_NEURONS = PIXELS;
    const auto OUTPUT_NEURONS = 100;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    net.addLayer(new DenseLayer(50));
    net.addLayer(new DenseLayer(50));
    net.addLayer(new DenseLayer(50));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = ImageNetDataLoader("val", ".ubyte", 11);
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    // Begin learning

    std::cout << std::format("eta = {} | HL = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
