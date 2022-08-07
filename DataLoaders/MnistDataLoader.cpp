#include "MnistDataLoader.h"

MnistDataLoader::MnistDataLoader(std::string dataFileName, std::string labelsFileName) {
    auto data = std::ifstream(dataFileName, std::ios::binary);
    auto labels = std::ifstream(labelsFileName, std::ios::binary);

    createDataItems(data, labels);
    resetDataIterator();
}

std::optional<int16_t> MnistDataLoader::loadDataItem(NeuralNetwork& neuralNetwork) {
    if (dataItemsIt == dataItems.end() || ++dataItemsIt == dataItems.end()) return {};

    for (int i = 0; i < dataItemsIt->pixels.size(); i++) {
        float value = dataItemsIt->pixels[i] / 255.0f; // [0, 1]
        neuralNetwork.setInputNode(i, value);
    }

    return dataItemsIt->label;
}

void MnistDataLoader::resetDataIterator() {
    dataItemsIt = dataItems.cbegin();
}

size_t MnistDataLoader::size() {
    return dataItems.size();
}

void MnistDataLoader::createDataItems(std::ifstream& data, std::ifstream& labels) {
    int32_t dataMagic, dataCount, dataRows, dataColumns;
    read<int32_t>(&dataMagic, data);
    read<int32_t>(&dataCount, data);
    read<int32_t>(&dataRows, data);
    read<int32_t>(&dataColumns, data);

    int32_t labelMagic, labelCount;
    read<int32_t>(&labelMagic, labels);
    read<int32_t>(&labelCount, labels);

    dataItems = MnistDataVector(dataCount);
    for (auto& dataItem : dataItems) {
        for (int i = 0; i < (dataRows * dataColumns); i++) { // for each pixel
            read<uint8_t>(&dataItem.pixels[i], data);
            /*if (nodeValue > 0.8) std::cout << "X";
            else if (nodeValue > 0.2) std::cout << "x";
            else std::cout << "_";
            if (i % dataRows == 0) std::cout << "\n";*/
        }
        //std::cout << "\n\n";
        read<uint8_t>(&dataItem.label, labels);
    }
}
