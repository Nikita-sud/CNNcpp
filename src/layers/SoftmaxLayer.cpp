#include "SoftmaxLayer.h"
#include <cmath>
#include <algorithm>

std::vector<std::vector<std::vector<double>>> SoftmaxLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    std::vector<double> flattenedInput = input[0][0];
    std::vector<double> softmaxOutput = softmax(flattenedInput);
    return { { softmaxOutput } };
}

std::vector<double> SoftmaxLayer::softmax(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    double max = *std::max_element(input.begin(), input.end());

    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max);
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> SoftmaxLayer::backward(const std::vector<std::vector<std::vector<double>>>& gradient) {
    return gradient;
}

std::vector<int> SoftmaxLayer::getOutputShape(const std::vector<int>& inputShape) {
    return { inputShape[0] };
}
