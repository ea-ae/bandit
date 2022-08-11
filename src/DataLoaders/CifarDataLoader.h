#pragma once
#include "DataLoader.h"
#include <array>
#include <vector>
#include <optional>
#include <string>
#include <memory>

constexpr int32_t CHANNELS = 3;
constexpr int32_t PIXELS_PC = 32 * 32; // * 3
constexpr int32_t PIXELS = CHANNELS * PIXELS_PC;

struct CifarDataItem {
    std::unique_ptr<std::array<uint8_t, PIXELS>> pixels = std::make_unique<std::array<uint8_t, PIXELS>>();
    uint16_t label;
};

using CifarDataVector = std::vector<CifarDataItem>;

class CifarDataLoader : public DataLoader {
private:
    std::unique_ptr<CifarDataVector> dataItems;
    CifarDataVector::const_iterator dataItemsIt;
public:
    CifarDataLoader(std::string dataFileName, size_t amount, bool useCoarseLabels = false);
    std::optional<int16_t> loadDataItem(NeuralNetwork& neuralNetwork, int32_t nthBatchItem);
    void resetDataIterator();
    size_t size();
private:
    void createDataItems(std::ifstream& data, bool useCoarseLabels);
};
