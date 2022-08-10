#include "mnist.h"
#include <iostream>
#include <format>
#include "DataLoaders/MnistDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "NeuralNetworks/Layers/DenseLayer.h"
#include "NeuralNetworks/Layers/ConvolutionalLayer.h"
#include "Trainers/ClassificationTrainer.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"
#include "NeuralNetworks/Neuron.h"

void mnist() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.02f; // default: 0.1-0.2
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto RELU_LEAK = 0.01f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    //net.addLayer(new ConvolutionalLayer(1, Size(28, 28), ...);
    net.addLayer(new ConvolutionalLayer(1, Size(3, 3), Size(2, 2), Size(1, 1), 1));
    net.addLayer(new DenseLayer(30));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = MnistDataLoader("./training-images.idx3-ubyte", "./training-labels.idx1-ubyte");
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = MnistDataLoader("./test-images.idx3-ubyte", "./test-labels.idx1-ubyte");
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);
    
    // Begin learning

    std::cout << std::format("eta = {} | batch = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, BATCH_SIZE, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
