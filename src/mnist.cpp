#include "mnist.h"

#include <format>
#include <iostream>

#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"
#include "DataLoaders/MnistDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "NeuralNetworks/Layers/ConvolutionalLayer.h"
#include "NeuralNetworks/Layers/DenseLayer.h"
#include "NeuralNetworks/Neurons/Neuron.h"
#include "Trainers/ClassificationTrainer.h"

void mnist() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.001f;      // default: 0.1-0.2
    const auto MOMENTUM_COEFFICIENT_MU = 0.9f;  // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.001f;  // no regularization: 0
    const auto RELU_LEAK = 0.01f;               // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    Layer* layer;
    layer = net.addLayer(new ConvolutionalLayer(Size3(28, 28, 1), Size3(3, 3, 50)));
    // layer = net.addLayer(new ConvolutionalLayer(layer->outputSize(), Size3(3, 3, 20)));
    // net.addLayer(new DenseLayer(50));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(&net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = MnistDataLoader("./training-images.idx3-ubyte", "./training-labels.idx1-ubyte");
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = MnistDataLoader("./test-images.idx3-ubyte", "./test-labels.idx1-ubyte");
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);

    // Begin learning

    std::cout << std::format("NN | eta = {} | batch = {} | lambda = {} | mu = {} | leak = {}\n",
                             LEARNING_RATE_ETA, BATCH_SIZE, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
