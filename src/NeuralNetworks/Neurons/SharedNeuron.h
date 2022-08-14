#pragma once
#include "Neuron.h"

class SharedNeuron : public Neuron {
   private:
    std::vector<Neuron*> inputNeurons;

   public:
    SharedNeuron(std::vector<Neuron*> inputNeurons, const ActivationFunction& activation, const CostFunction& cost,
                 std::vector<std::shared_ptr<Weight>>* sharedWeights, Bias* sharedBias);

   private:
    Neuron& getInputNeuron(size_t i);
    size_t getInputNeuronCount();
};
