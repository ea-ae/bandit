#include "ImageNetDataLoader.h"
#include <iostream>

ImageNetDataLoader::ImageNetDataLoader(std::string filePrefix, std::string fileSuffix, int32_t count) {
    count = 1; // temp
    for (int i = 0; i < count; i++) {
        std::string fileName = filePrefix + std::to_string(i) + fileSuffix;
        auto data = std::ifstream(fileName, std::ios::binary);
        createDataItems(data);
    }

    resetDataIterator();
}

std::optional<int16_t> ImageNetDataLoader::loadDataItem(NeuralNetwork& neuralNetwork, int32_t nthBatchItem) {
    if (dataItemsIt == dataItems->end() || ++dataItemsIt == dataItems->end()) return {};

    for (int i = 0; i < dataItemsIt->pixels->size(); i++) {
        auto pixels = *dataItemsIt->pixels.get();
        float value = pixels[i] / 255.0f; // [0, 1]
        neuralNetwork.setInputNode(i, nthBatchItem, value);
    }

    return dataItemsIt->label;
}

void ImageNetDataLoader::resetDataIterator() {
    dataItemsIt = dataItems->cbegin();
}

size_t ImageNetDataLoader::size() {
    return dataItems->size();
}

void ImageNetDataLoader::createDataItems(std::ifstream& data) {
    int32_t magic;
    read<int32_t>(&magic, data);
    assert(magic == MAGIC);

    while (true) {
        read<int32_t>(&magic, data);
        assert(magic == MAGIC || magic == 0);
        if (magic == 0) break; // EoF

        dataItems->emplace_back();
        ImageNetDataItem& dataItem = dataItems->back();
        read<uint16_t>(&dataItem.label, data);
        auto pixels = *dataItem.pixels.get();
        for (int i = 0; i < dataItem.pixels->size(); i++) {
            read<uint8_t>(&pixels[i], data);
        }
    }
}
