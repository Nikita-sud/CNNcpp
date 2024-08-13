#include "cnn/MNISTReader.h"
#include <iostream>
#include <random>
#include <algorithm>

int main() {
    try {
        std::string imagesFile = "../data/train-images.idx3-ubyte";
        std::string labelsFile = "../data/train-labels.idx1-ubyte";
        std::vector<ImageData> dataset = MNISTReader::readMNISTData(imagesFile, labelsFile);

        // Shuffle the dataset
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(dataset.begin(), dataset.end(), g);

        // Print the first image and label
        const auto& imageData = dataset[0].getImageData();
        const auto& label = dataset[0].getLabel();

        for (const auto& row : imageData[0]) {
            for (double pixel : row) {
                std::cout << (pixel > 0.5 ? "*" : " ");
            }
            std::cout << "\n";
        }
        std::cout << "Label: " << std::distance(label.begin(), std::max_element(label.begin(), label.end())) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}