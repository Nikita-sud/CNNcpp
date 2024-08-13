#include "layers/FlattenLayer.h"
#include <stdexcept>

FlattenLayer::FlattenLayer() : depth(0), height(0), width(0) {}

void FlattenLayer::initialize(const std::vector<int>& inputShape) {
    if (inputShape.size() != 3) {
        throw std::invalid_argument("Expected input shape with 3 dimensions (depth, height, width).");
    }
    depth = inputShape[0];
    height = inputShape[1];
    width = inputShape[2];
}

std::vector<std::vector<std::vector<double>>> FlattenLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    if (input.size() != depth || input[0].size() != height || input[0][0].size() != width) {
        throw std::invalid_argument("Input dimensions do not match the initialized shape.");
    }

    std::vector<std::vector<std::vector<double>>> output(1, std::vector<std::vector<double>>(1, std::vector<double>(depth * height * width)));

    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                output[0][0][d * height * width + i * width + j] = input[d][i][j];
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> FlattenLayer::backward(std::vector<std::vector<std::vector<double>>> gradient) {
    if (gradient[0][0].size() != depth * height * width) {
        throw std::invalid_argument("Gradient dimensions do not match the expected shape.");
    }

    std::vector<std::vector<std::vector<double>>> reshapedGradient(depth, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                reshapedGradient[d][i][j] = gradient[0][0][d * height * width + i * width + j];
            }
        }
    }

    return reshapedGradient;
}

std::vector<int> FlattenLayer::getOutputShape(const std::vector<int>& inputShape) {
    if (inputShape.size() != 3) {
        throw std::invalid_argument("Expected input shape with 3 dimensions (depth, height, width).");
    }
    int flatSize = inputShape[0] * inputShape[1] * inputShape[2];
    return {flatSize};
}