﻿#include "mnist.h"
#include <iostream>
#include <format>
#include "NeuralNetworks/Neuron.h"
#include "DataLoaders/MnistDataLoader.h"
#include "Trainers/ClassificationTrainer.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"

void mnist() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.2f; // default: 0.1
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto RELU_LEAK = 0.01f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;
    const auto HIDDEN_LAYERS = std::vector<int32_t>{ 300 }; // default: 300

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(activationFunction, costFunction,
        INPUT_NEURONS, OUTPUT_NEURONS, HIDDEN_LAYERS);
    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA, BATCH_SIZE);

    // Load datasets

    auto trainingDataSet = MnistDataLoader("./training-images.idx3-ubyte", "./training-labels.idx1-ubyte");
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = MnistDataLoader("./test-images.idx3-ubyte", "./test-labels.idx1-ubyte");
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);
    
    // Begin learning

    std::cout << std::format("eta = {} | batch = {} | HL = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, BATCH_SIZE, trainer.getHiddenLayersStatusMessage(HIDDEN_LAYERS), 
        REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
