#pragma once
#include <string>
#include <vector>

#include "../DataLoaders/DataLoader.h"

enum class DataSourceType {
    Training,
    Validation,  // prefer k-fold cross-validation
    Testing
};

class Trainer {
   protected:
    std::vector<DataLoader*> trainingDataSources = std::vector<DataLoader*>();
    // std::vector<DataSource> validationDataSources = std::vector<DataSource>();
    std::vector<DataLoader*> testingDataSources = std::vector<DataLoader*>();

   public:
    void addDataSource(DataLoader* dataSource, DataSourceType dataSourceType);
    static std::string getHiddenLayersStatusMessage(std::vector<int32_t> hiddenLayers);

   protected:
    std::string getEpochStatusMessage();
};
