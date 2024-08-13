#ifndef MNIST_READER_H
#define MNIST_READER_H

#include "utils/ImageData.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>

class MNISTReader {
public:
    static std::vector<ImageData> readMNISTData(const std::string& imagesFile, const std::string& labelsFile);

private:
    static int32_t readInt(std::ifstream& stream);
};

#endif // MNIST_READER_H