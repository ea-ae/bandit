#pragma once
#include "Trainer.h"
#include "../NeuralNetworks/ClassificationNeuralNetwork.h"

class ClassificationTrainer : public Trainer {
public:
    ClassificationTrainer(ClassificationNeuralNetwork& net, float learningRate);
    void train();
private:
    ClassificationNeuralNetwork& net;
    const float learningRate;
};
