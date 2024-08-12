#include "ImageData.h"

ImageData::ImageData(const std::vector<std::vector<std::vector<double>>>& imageData, const std::vector<double>& label)
    : imageData(imageData), label(label) {}

const std::vector<std::vector<std::vector<double>>>& ImageData::getImageData() const {
    return imageData;
}

const std::vector<double>& ImageData::getLabel() const {
    return label;
}