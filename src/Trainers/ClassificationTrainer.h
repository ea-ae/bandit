#pragma once
#include "../NeuralNetworks/ClassificationNeuralNetwork.h"
#include "Trainer.h"

class ClassificationTrainer : public Trainer {
   public:
    ClassificationTrainer(ClassificationNeuralNetwork* net, float learningRate);
    void train();

   private:
    ClassificationNeuralNetwork* net;
    const float learningRate;
};
