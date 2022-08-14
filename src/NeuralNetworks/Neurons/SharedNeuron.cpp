#include "SharedNeuron.h"

SharedNeuron::SharedNeuron(std::vector<Neuron*> inputNeurons, const ActivationFunction& activation, const CostFunction& cost,
                           std::vector<std::shared_ptr<Weight>>* sharedWeights, Bias* sharedBias)
    : Neuron(activation, cost), inputNeurons(inputNeurons) {
    std::copy(sharedWeights->begin(), sharedWeights->end(), std::back_inserter(weights));

    for (auto& weight : weights) {
        weight->weight = activationFunction.generateRandomWeight(inputNeurons.size());  // initialize weights (todo: only once!!)
    }
    bias = std::shared_ptr<Bias>(sharedBias);
}

Neuron& SharedNeuron::getInputNeuron(size_t i) {
    return *inputNeurons[i];
}

size_t SharedNeuron::getInputNeuronCount() {
    return inputNeurons.size();
}
