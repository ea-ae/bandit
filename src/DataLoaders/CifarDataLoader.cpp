#include "CifarDataLoader.h"

CifarDataLoader::CifarDataLoader(std::string dataFileName, size_t amount, bool useCoarseLabels) 
    : dataItems(std::make_unique<CifarDataVector>(amount)) {
    auto data = std::ifstream(dataFileName, std::ios::binary);

    createDataItems(data, useCoarseLabels);
    resetDataIterator();
}

std::optional<int16_t> CifarDataLoader::loadDataItem(NeuralNetwork& neuralNetwork, int32_t nthBatchItem) {
    if (dataItemsIt == dataItems->end() || ++dataItemsIt == dataItems->end()) return {};

    for (int i = 0; i < dataItemsIt->pixels->size(); i++) {
        auto pixels = *dataItemsIt->pixels.get();
        float value = static_cast<float>(pixels[i]) / 255.0f; // [0, 1]
        neuralNetwork.setInputNode(i, nthBatchItem, value);
    }

    return dataItemsIt->label;
}

void CifarDataLoader::resetDataIterator() {
    dataItemsIt = dataItems->cbegin();
}

size_t CifarDataLoader::size() {
    return dataItems->size();
}

void CifarDataLoader::createDataItems(std::ifstream& data, bool useCoarseLabels) {
    for (auto& dataItem : *dataItems) {
        int8_t coarseLabel, fineLabel;
        read<int8_t>(&coarseLabel, data);
        read<int8_t>(&fineLabel, data);

        dataItem.label = useCoarseLabels ? coarseLabel : fineLabel;
        // std::cout << static_cast<int32_t>(coarseLabel) << " and " << static_cast<int32_t>(fineLabel) << "\n";

        auto& pixels = *dataItem.pixels.get();
        for (size_t p = 0; p < PIXELS_PC; p++) {
            for (size_t c = 0; c < CHANNELS; c++) {
                read<uint8_t>(&pixels[p + c * PIXELS_PC], data); // realign RGB pixels
                //std::cout << "got " << p + c * PIXELS_PC << " w " << (int)pixels[p + c * PIXELS_PC] << "\n";
            }
        }
    }
}
