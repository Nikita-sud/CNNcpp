#include "layers/ConvolutionalLayer.h"
#include "utils/MatrixUtils.h"
#include "utils/activationFunctions/ReLU.h"
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters, int stride, std::shared_ptr<ActivationFunction> activationFunction)
    : filterSize(filterSize), numFilters(numFilters), stride(stride), activationFunction(activationFunction) {}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters, std::shared_ptr<ActivationFunction> activationFunction)
    : ConvolutionalLayer(filterSize, numFilters, 1, activationFunction) {}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters)
    : ConvolutionalLayer(filterSize, numFilters, 1, std::make_shared<ReLU>()) {}

void ConvolutionalLayer::initializeFilters(int inputDepth) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std::sqrt(2.0 / (inputDepth * filterSize * filterSize)));

    filters.resize(numFilters);
    for (int f = 0; f < numFilters; ++f) {
        filters[f].resize(inputDepth);
        for (int d = 0; d < inputDepth; ++d) {
            filters[f][d].resize(filterSize, std::vector<double>(filterSize));
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    filters[f][d][i][j] = dist(gen);
                }
            }
        }
    }
}

void ConvolutionalLayer::initializeBiases() {
    biases.resize(numFilters);
    std::generate(biases.begin(), biases.end(), std::rand);
}

void ConvolutionalLayer::initializeAccumulatedGradients() {
    accumulatedFilterGradients.resize(numFilters);
    for (int f = 0; f < numFilters; ++f) {
        accumulatedFilterGradients[f].resize(filters[f].size());
        for (size_t d = 0; d < filters[f].size(); ++d) {
            accumulatedFilterGradients[f][d].resize(filterSize, std::vector<double>(filterSize, 0.0));
        }
    }
    accumulatedBiasGradients.resize(numFilters, 0.0);
}

void ConvolutionalLayer::initialize(const std::vector<int>& inputShape) {
    int inputDepth = inputShape[0];
    initializeFilters(inputDepth);
    initializeBiases();
    initializeAccumulatedGradients();
}

std::vector<std::vector<std::vector<double>>> ConvolutionalLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    int inputDepth = input.size();
    int inputSize = input[0].size();
    int outputSize = (inputSize - filterSize) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(numFilters, std::vector<std::vector<double>>(outputSize, std::vector<double>(outputSize)));
    activatedOutput = std::vector<std::vector<std::vector<double>>>(numFilters, std::vector<std::vector<double>>(outputSize, std::vector<double>(outputSize)));

    for (int f = 0; f < numFilters; ++f) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                int x = i * stride;
                int y = j * stride;
                double sum = 0.0;
                for (int d = 0; d < inputDepth; ++d) {
                    sum += MatrixUtils::applyFilter(input[d], filters[f][d], x, y);
                }
                output[f][i][j] = sum + biases[f];
                activatedOutput[f][i][j] = activationFunction->activate(output[f][i][j]);
            }
        }
    }

    return activatedOutput;
}

std::vector<std::vector<std::vector<double>>> ConvolutionalLayer::backward(std::vector<std::vector<std::vector<double>>> gradient) {
    if (gradient.empty() || input.empty() || activatedOutput.empty()) {
        throw std::runtime_error("Invalid input: one or more vectors are empty");
    }

    int inputDepth = input.size();
    int inputSize = input[0].size();
    int outputSize = activatedOutput[0].size();
    std::vector<std::vector<std::vector<double>>> inputGradient(inputDepth, std::vector<std::vector<double>>(inputSize, std::vector<double>(inputSize)));

    // Backpropagation through activation function
    for (int f = 0; f < numFilters; ++f) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                gradient[f][i][j] *= activationFunction->derivative(activatedOutput[f][i][j]);
            }
        }
    }

    // Calculate gradients for filters and inputs
    for (int f = 0; f < numFilters; ++f) {
        for (int d = 0; d < inputDepth; ++d) {
            // Calculate gradient for filters
            auto filterGrad = MatrixUtils::convolve(input[d], gradient[f], stride);
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    accumulatedFilterGradients[f][d][i][j] += filterGrad[i][j];
                }
            }

            // Calculate gradient for input
            auto rotatedFilter = MatrixUtils::rotate180(filters[f][d]);
            auto inputGrad = MatrixUtils::fullConvolve(rotatedFilter, gradient[f]);
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    inputGradient[d][i][j] += inputGrad[i][j];
                }
            }
        }

        // Calculate gradient for biases
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                accumulatedBiasGradients[f] += gradient[f][i][j];
            }
        }
    }

    return inputGradient;
}

void ConvolutionalLayer::updateParameters(double learningRate, int miniBatchSize) {
    for (int f = 0; f < numFilters; ++f) {
        for (int d = 0; d < filters[f].size(); ++d) {
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    filters[f][d][i][j] -= learningRate * accumulatedFilterGradients[f][d][i][j] / miniBatchSize;
                    accumulatedFilterGradients[f][d][i][j] = 0;
                }
            }
        }
        biases[f] -= learningRate * accumulatedBiasGradients[f] / miniBatchSize;
        accumulatedBiasGradients[f] = 0;
    }
}

void ConvolutionalLayer::resetGradients() {
    for (int f = 0; f < numFilters; ++f) {
        for (int d = 0; d < filters[f].size(); ++d) {
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    accumulatedFilterGradients[f][d][i][j] = 0;
                }
            }
        }
        accumulatedBiasGradients[f] = 0;
    }
}

std::vector<int> ConvolutionalLayer::getOutputShape(const std::vector<int>& inputShape) {
    int inputSize = inputShape[1];
    int outputSize = (inputSize - filterSize) / stride + 1;
    return {numFilters, outputSize, outputSize};
}