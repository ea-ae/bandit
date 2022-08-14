#pragma once
#include "Neuron.h"

class DenseNeuron : public Neuron {
private:
    std::vector<std::unique_ptr<Neuron>>* inputNeurons;
public:
    DenseNeuron(std::vector<std::unique_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost);
private:
    Neuron& getInputNeuron(size_t i);
    size_t getInputNeuronCount();
};
