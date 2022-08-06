#pragma once
#include "DataLoader.h"
#include <array>
#include <optional>
#include <fstream>
#include <string>
#include <vector>
#include "../NeuralNetworks/NeuralNetwork.h"

struct MnistDataItem {
    std::array<uint8_t, 784> pixels;
    uint8_t label;
};

using MnistDataVector = std::vector<MnistDataItem>;

class MnistDataLoader : public DataLoader {
private:
    MnistDataVector dataItems;
    MnistDataVector::const_iterator dataItemsIt;
public:
    MnistDataLoader(std::string dataFileName, std::string labelsFileName);
    std::optional<int8_t> loadDataItem(NeuralNetwork& neuralNetwork);
    void resetDataIterator();
    size_t size();
private:
    void createDataItems(std::ifstream& data, std::ifstream& labels);
    template <class T> static void endswap(T* objp);
    template <class T> static void read(T* buffer, std::ifstream& stream);
};

#include "MnistDataLoader.tpp"
