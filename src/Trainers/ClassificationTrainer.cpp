#include "ClassificationTrainer.h"

#include <chrono>
#include <cmath>
#include <format>
#include <iostream>

#include "../NeuralNetworks/Neurons/Neuron.h"
#include "../bandit.h"

using namespace std::chrono;

ClassificationTrainer::ClassificationTrainer(ClassificationNeuralNetwork& net, float learningRate)
    : net(net), learningRate(learningRate) {
    std::cout << "NN | Initializing classification trainer\n";
}

void ClassificationTrainer::train() {
    // Begin learning

    int32_t epoch = 1;
    auto trainingStart = steady_clock::now();
    auto trainingDataSet = trainingDataSources[0];  // OK for now
    // auto testingDataSet = trainingDataSources[0]; // !
    auto testingDataSet = testingDataSources[0];

    while (true) {
        auto epochStart = steady_clock::now();
        int32_t trainingCorrect = 0, testsDone = 0, testingCorrect = 0;
        auto dataSetRatio = static_cast<int32_t>(
            std::ceilf(static_cast<float>(trainingDataSet->size()) / testingDataSet->size()));

        // Training and testing

        BatchLabelArray batchLabels = BatchLabelArray(BatchLabelArray::Zero());
        std::optional<int8_t> label;
        bool trainingDataLeft = true, testingDataLeft = true;
        while (true) {
            if (trainingDataLeft) {
                // occasionally switch between training/testing
                for (int32_t batchesDone = 0; batchesDone < dataSetRatio; batchesDone++) {
                    if (!trainingDataLeft) {
                        std::cout << "Ran out of training data on point\n";
                        break;
                    }

                    for (int32_t batchItemsDone = 0; batchItemsDone < BATCH_SIZE; batchItemsDone++) {
                        label = trainingDataSet->loadDataItem(net, batchItemsDone);
                        if (!label.has_value()) {
                            trainingDataLeft = false;
                            break;
                        }
                        batchLabels[batchItemsDone] = label.value();
                    }

                    if (!trainingDataLeft) {  // make sure we got a full batch of labels and inputs
                        break;
                    }

                    net.calculateOutput();
                    net.backpropagate(batchLabels);
                    for (int32_t i = 0; i < batchLabels.size(); i++) {
                        if (net.getHighestOutputNode(i) == batchLabels[i]) trainingCorrect++;
                    }
                    net.update(learningRate);
                }
            }

            if (testingDataLeft) {
                for (int32_t batchItemsDone = 0; batchItemsDone < BATCH_SIZE; batchItemsDone++) {
                    label = testingDataSet->loadDataItem(net, batchItemsDone);
                    testsDone++;
                    if (!label.has_value()) {
                        testingDataLeft = false;
                        break;
                    }
                    batchLabels[batchItemsDone] = label.value();
                }

                if (trainingDataLeft) {  // make sure we got a full batch of labels and inputs
                    net.calculateOutput();
                    for (int32_t i = 0; i < batchLabels.size(); i++) {
                        if (net.getHighestOutputNode(i) == batchLabels[i]) testingCorrect++;
                    }
                }
            }

            if (!trainingDataLeft && !testingDataLeft) break;

            float testingProgress = 100.0f * testsDone / testingDataSet->size();
            auto progressBar = std::string(static_cast<int32_t>(std::floor(testingProgress * 0.3)), '=');
            std::cout << std::format("\rNN | Epoch {:03} | {:05.2f}% done | {:30} |",
                                     epoch, testingProgress, progressBar);
        }

        // Iterate over the same data again on next epoch

        trainingDataSet->resetDataIterator();
        testingDataSet->resetDataIterator();

        // Calculate and print stats

        float trainPassRate = static_cast<float>(100 * trainingCorrect) / trainingDataSet->size();
        float testPassRate = static_cast<float>(100 * testingCorrect) / testingDataSet->size();

        auto epochEnd = steady_clock::now();
        auto epochDuration = duration_cast<seconds>(epochEnd - epochStart).count();
        auto totalDuration = duration_cast<seconds>(epochEnd - trainingStart).count();

        std::cout << std::format("\rNN | Epoch {:03} | training: {:.2f}%, testing: {:.2f}% | took: {}s, total: {}s\n",
                                 epoch, trainPassRate, testPassRate, epochDuration, totalDuration);

        epoch++;
    }
}
