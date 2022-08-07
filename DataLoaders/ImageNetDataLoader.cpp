#include "ImageNetDataLoader.h"
#include <cassert>
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

        auto dataItem = ImageNetDataItem();
        read<uint16_t>(&dataItem.label, data);
        auto pixels = *dataItem.pixels.get();
        for (int i = 0; i < dataItem.pixels->size(); i++) {
            read<uint8_t>(&pixels[i], data);
        }
        dataItems->push_back(std::move(dataItem));
        std::cout << "added\n";
    }
}
