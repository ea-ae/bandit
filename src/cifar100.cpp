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

    const auto LEARNING_RATE_ETA = 0.001f; // default: 0.1-0.2
    const auto MOMENTUM_COEFFICIENT_MU = 0.0f; // no momentum: 0
    const auto REGULARIZATION_LAMBDA = 0.0f; // no regularization: 0, default: 0.001
    const auto RELU_LEAK = 0.01f; // no leak: 0

    const auto USE_COARSE_LABELS = true;
    const auto INPUT_NEURONS = 32 * 32 * 3;
    const auto OUTPUT_NEURONS = USE_COARSE_LABELS ? 10 : 100;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network and trainer

    auto net = ClassificationNeuralNetwork(INPUT_NEURONS, OUTPUT_NEURONS);
    net.addLayer(new ConvolutionalLayer(Size(32, 32), Size(8, 8), Size(3, 3), 40, 1, 3));
    // net.addLayer(new DenseLayer(30));
    net.buildLayers(activationFunction, costFunction);

    auto trainer = ClassificationTrainer(net, LEARNING_RATE_ETA);

    // Load datasets

    auto trainingDataSet = CifarDataLoader("./train.bin", 50000, true);
    trainer.addDataSource(&trainingDataSet, DataSourceType::Training);

    auto testingDataSet = CifarDataLoader("./test.bin", 10000, true);
    trainer.addDataSource(&testingDataSet, DataSourceType::Testing);

    // Begin learning

    std::cout << std::format("eta = {} | batch = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, BATCH_SIZE, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    trainer.train();
}
