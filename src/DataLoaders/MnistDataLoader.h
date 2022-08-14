#pragma once
#include <array>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "DataLoader.h"

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
    std::optional<int16_t> loadDataItem(NeuralNetwork& neuralNetwork, int32_t nthBatchItem);
    void resetDataIterator();
    size_t size();

   private:
    void createDataItems(std::ifstream& data, std::ifstream& labels);
};
