#include "Trainer.h"
#include <iostream>
#include <format>
#include <sstream>

void Trainer::addDataSource(DataLoader* dataSource, DataSourceType dataSourceType) {
    std::string type;
    switch (dataSourceType) {
    case DataSourceType::Training:
        trainingDataSources.push_back(dataSource);
        type = "training";
        break;
    case DataSourceType::Testing:
        testingDataSources.push_back(dataSource);
        type = "testing";
        break;
    case DataSourceType::Validation:
        throw std::runtime_error("Validation data type is unsupported");
    }

    std::cout << std::format("DS | Finished reading {} {} data items into memory\n", dataSource->size(), type);
    // std::cout << "Finished reading " << dataSource->size() << " " << type << " data items into memory\n";
}

std::string Trainer::getHiddenLayersStatusMessage(std::vector<int32_t> hiddenLayers) {
    std::stringstream hlStringStream;
    std::string hlString = "0";
    if (hiddenLayers.size() > 0) {
        std::copy(hiddenLayers.begin(), hiddenLayers.end(), 
            std::ostream_iterator<int32_t>(hlStringStream, "x"));
        hlString = hlStringStream.str();
        hlString.pop_back();
    }
    return hlString;
}

std::string Trainer::getEpochStatusMessage() {
    return std::string();
}
