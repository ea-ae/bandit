#include "DenseNeuron.h"

DenseNeuron::DenseNeuron(std::vector<std::unique_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost)
    : Neuron(activation, cost), inputNeurons(inputNeurons) {
    if (inputNeurons != nullptr) {
        weights = std::vector<std::shared_ptr<Weight>>(inputNeurons->size());

        std::generate(weights.begin(), weights.end(), [&]() {  // initialize weights
            Weight* weight = new Weight();
            weight->weight = activationFunction.generateRandomWeight(inputNeurons->size());
            return std::shared_ptr<Weight>(weight);
        });
        bias = std::make_shared<Bias>();
    }
}

Neuron& DenseNeuron::getInputNeuron(size_t i) {
    return *(*inputNeurons)[i].get();
}

size_t DenseNeuron::getInputNeuronCount() {
    return inputNeurons->size();
}
