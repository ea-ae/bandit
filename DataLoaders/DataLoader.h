#pragma once
#include <fstream>

class DataLoader {
protected:
    template <class T> static void endswap(T* objp);
    template <class T> static void read(T* buffer, std::ifstream& stream, bool littleEndian = false);
};

#include "DataLoader.tpp"
