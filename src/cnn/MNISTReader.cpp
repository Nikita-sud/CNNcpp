#include "cnn/MNISTReader.h"
#include <iostream>
#include <algorithm> // For std::shuffle
#include <random>    // For random number generation

std::vector<ImageData> MNISTReader::readMNISTData(const std::string& imagesFile, const std::string& labelsFile) {
    std::ifstream images(imagesFile, std::ios::binary);
    std::ifstream labels(labelsFile, std::ios::binary);

    if (!images.is_open() || !labels.is_open()) {
        throw std::runtime_error("Failed to open MNIST files.");
    }

    // Read magic numbers and sizes
    int32_t magicNumberImages = readInt(images);
    int32_t numberOfImages = readInt(images);
    int32_t rows = readInt(images);
    int32_t cols = readInt(images);

    int32_t magicNumberLabels = readInt(labels);
    int32_t numberOfLabels = readInt(labels);

    if (numberOfImages != numberOfLabels) {
        throw std::runtime_error("Number of images and labels do not match.");
    }

    std::vector<ImageData> dataset;

    for (int i = 0; i < numberOfImages; ++i) {
        // Read and normalize image data
        std::vector<std::vector<std::vector<double>>> imageData(1, std::vector<std::vector<double>>(rows, std::vector<double>(cols)));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                uint8_t pixel = 0;
                images.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                imageData[0][r][c] = pixel / 255.0;
            }
        }

        // Read label
        uint8_t label = 0;
        labels.read(reinterpret_cast<char*>(&label), sizeof(label));
        std::vector<double> arrayLabel(10, 0.0);
        arrayLabel[label] = 1.0;

        // Store the image data and label in the dataset
        dataset.emplace_back(imageData, arrayLabel);
    }

    return dataset;
}

int32_t MNISTReader::readInt(std::ifstream& stream) {
    int32_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    // Convert from big endian to little endian if necessary
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}