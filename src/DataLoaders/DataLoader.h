#pragma once
#include <fstream>

#include "../NeuralNetworks/NeuralNetwork.h"

class DataLoader {
   public:
    virtual ~DataLoader() = default;
    virtual std::optional<int16_t> loadDataItem(NeuralNetwork& neuralNetwork, int32_t nthBatchItem) = 0;
    virtual void resetDataIterator() = 0;
    virtual size_t size() = 0;

   protected:
    template <class T>
    static void endswap(T* objp);
    template <class T>
    static void read(T* buffer, std::ifstream& stream, bool littleEndian = false);
};

#include "DataLoader.tpp"
