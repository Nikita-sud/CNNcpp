#include "layers/ConvolutionalLayer.h"
#include "utils/MatrixUtils.h"
#include "utils/activationFunctions/ReLU.h"

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters, int stride, ActivationFunction* activationFunction)
    : filterSize(filterSize), numFilters(numFilters), stride(stride), activationFunction(activationFunction) {}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters, ActivationFunction* activationFunction)
    : ConvolutionalLayer(filterSize, numFilters, 1, activationFunction) {}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters)
    : ConvolutionalLayer(filterSize, numFilters, 1, new ReLU()) {}


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

std::vector<std::vector<std::vector<double>>> ConvolutionalLayer::backward(const std::vector<std::vector<std::vector<double>>>& gradient) {
    int inputDepth = input.size();
    int inputSize = input[0].size();
    int outputSize = activatedOutput[0].size();
    std::vector<std::vector<std::vector<double>>> inputGradient(inputDepth, std::vector<std::vector<double>>(inputSize, std::vector<double>(inputSize)));

    for (int f = 0; f < numFilters; ++f) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                double grad = gradient[f][i][j] * activationFunction->derivative(activatedOutput[f][i][j]);
                std::vector<std::vector<double>> gradMatrix = {{grad}}; // Wrap grad in a matrix
                for (int d = 0; d < inputDepth; ++d) {
                    // Correctly apply the convolve and fullConvolve functions with the correct types
                    auto filterGrad = MatrixUtils::convolve(input[d], gradMatrix, stride);
                    for (int fi = 0; fi < filterSize; ++fi) {
                        for (int fj = 0; fj < filterSize; ++fj) {
                            accumulatedFilterGradients[f][d][fi][fj] += filterGrad[fi][fj];
                        }
                    }
                    auto inputGrad = MatrixUtils::fullConvolve(MatrixUtils::rotate180(filters[f][d]), gradMatrix);
                    for (int ii = 0; ii < inputSize; ++ii) {
                        for (int jj = 0; jj < inputSize; ++jj) {
                            inputGradient[d][ii][jj] += inputGrad[ii][jj];
                        }
                    }
                }
                accumulatedBiasGradients[f] += grad;
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
