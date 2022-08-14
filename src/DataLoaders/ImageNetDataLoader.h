#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataLoader.h"

constexpr int32_t MAGIC = (41 << 24) + (42 << 16) + (43 << 8) + 44;
constexpr int32_t PIXELS = 224 * 224 * 3;

struct ImageNetDataItem {
    std::unique_ptr<std::array<uint8_t, PIXELS>> pixels = std::make_unique<std::array<uint8_t, PIXELS>>();
    uint16_t label;
};

using ImageNetDataVector = std::vector<ImageNetDataItem>;

class ImageNetDataLoader : public DataLoader {
   private:
    std::unique_ptr<ImageNetDataVector> dataItems = std::make_unique<ImageNetDataVector>();
    ImageNetDataVector::const_iterator dataItemsIt;

   public:
    ImageNetDataLoader(std::string filePrefix, std::string fileSuffix, int32_t count);
    std::optional<int16_t> loadDataItem(NeuralNetwork* neuralNetwork, int32_t nthBatchItem);
    void resetDataIterator();
    size_t size();

   private:
    void createDataItems(std::ifstream& data);
};
