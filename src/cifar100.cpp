#include "cifar100.h"
#include <iostream>
#include <format>
#include "DataLoaders/CifarDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "NeuralNetworks/Layers/DenseLayer.h"
#include "NeuralNetworks/Layers/ConvolutionalLayer.h"
#include "Trainers/ClassificationTrainer.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"
#include "NeuralNetworks/Neurons/Neuron.h"

void cifar100() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.0005f; // default: 0.1-0.2
    const auto MOMENTUM_COEFFICIENT_MU = 0.8f; // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0, default: 0.001
    const auto RELU_LEAK = 0.01f; // no leak: 0

    const auto USE_COARSE_LABELS = true;
    const auto INPUT_NEURONS = 32 * 32 * 3;
    //const auto INPUT_NEURONS = 3 * 3 * 2;
    const auto OUTPUT_NEURONS = USE_COARSE_LABELS ? 20 : 100;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    // todo instead of [Input]Size() + channels, create 2DSize() and 3DSize() or use eigen
    //net.addLayer(new ConvolutionalLayer(Size(3, 3), Size(2, 2), Size(1, 1), 1, 2));
    Layer* layer;
    layer = net.addLayer(new ConvolutionalLayer(Size3(32, 32, 3), Size3(4, 4, 20), Size(2, 2)));
    //layer = net.addLayer(new ConvolutionalLayer(layer->outputSize(), Size3(4, 4, 10), Size(2, 2)));
    //layer = net.addLayer(new ConvolutionalLayer(layer->outputSize(), Size3(5, 5, 5), Size(2, 2)));
    //layer = net.addLayer(new ConvolutionalLayer(layer->outputSize(), Size3(4, 4, 5)));
    net.addLayer(new DenseLayer(50));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = CifarDataLoader("./train.bin", 50000, USE_COARSE_LABELS);
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = CifarDataLoader("./test.bin", 10000, USE_COARSE_LABELS);
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);

    // Begin learning

    std::cout << std::format("NN | eta = {} | batch = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, BATCH_SIZE, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
