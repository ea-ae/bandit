#pragma once

template <class T>
static void DataLoader::endswap(T* objp) {
    unsigned char* memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}

template <class T>
static void DataLoader::read(T* buffer, std::ifstream& stream, bool littleEndian) {
    stream.read(reinterpret_cast<char*>(buffer), sizeof(T));
    if (!littleEndian) endswap(buffer);
}
