#include "ClassificationTrainer.h"
#include <chrono>
#include <iostream>
#include <format>

using namespace std::chrono;

ClassificationTrainer::ClassificationTrainer(ClassificationNeuralNetwork& net, float learningRate, int32_t batchSize) 
    : net(net), learningRate(learningRate), batchSize(batchSize) {}

void ClassificationTrainer::train() {
    // Begin learning

    int32_t epoch = 1;
    auto trainingStart = steady_clock::now();
    auto trainingDataSet = trainingDataSources[0]; // OK for now
    // auto testingDataSet = trainingDataSources[0]; // !
    auto testingDataSet = testingDataSources[0];

    while (true) {
        auto epochStart = steady_clock::now();
        int32_t trainingCorrect = 0, testingCorrect = 0;
        auto dataSetRatio = static_cast<int32_t>(
            std::ceilf(static_cast<float>(trainingDataSet->size()) / testingDataSet->size()));

        // Training and testing

        std::optional<int8_t> trainLabel, testLabel;
        while (true) {
            int32_t batchItemsDone = 0, batchesDone = 0;

            while ((trainLabel = trainingDataSet->loadDataItem(net)).has_value()) {
                net.calculateOutput();
                net.backpropagate(trainLabel.value());

                if (net.getHighestOutputNode() == trainLabel.value()) trainingCorrect++;

                if (++batchItemsDone == batchSize) {
                    batchItemsDone = 0;
                    net.update(batchSize, learningRate);
                    if (++batchesDone == dataSetRatio) break; // occasionally switch between training/testing
                }
            }

            while ((testLabel = testingDataSet->loadDataItem(net)).has_value()) {
                net.calculateOutput();
                if (net.getHighestOutputNode() == testLabel) testingCorrect++;

                if (++batchItemsDone == batchSize) break; // do 1 testing batch for every n training batches
            }

            if (!trainLabel.has_value() && !testLabel.has_value()) break; // both data sets depleted
        }

        // Iterate over same data again on next epoch

        trainingDataSet->resetDataIterator();
        testingDataSet->resetDataIterator();

        // Calculate and print stats

        float trainPassRate = static_cast<float>(100 * trainingCorrect) / trainingDataSet->size();
        float testPassRate = static_cast<float>(100 * testingCorrect) / testingDataSet->size();

        auto epochEnd = steady_clock::now();
        auto epochDuration = duration_cast<seconds>(epochEnd - epochStart).count();
        auto totalDuration = duration_cast<seconds>(epochEnd - trainingStart).count();

        std::cout << std::format("Epoch {:03} | training: {:.2f}%, testing: {:.2f}% | took: {}s, total: {}s\n",
            epoch, trainPassRate, testPassRate, epochDuration, totalDuration);

        epoch++;
    }
}
