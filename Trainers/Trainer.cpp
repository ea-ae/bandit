#include "Trainer.h"
#include <sstream>

void Trainer::addDataSource(DataLoader* dataSource, DataSourceType dataSourceType) {
    switch (dataSourceType) {
    case DataSourceType::Training:
        trainingDataSources.push_back(dataSource);
        break;
    case DataSourceType::Testing:
        testingDataSources.push_back(dataSource);
    }
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
