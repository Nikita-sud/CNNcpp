#include "layers/FullyConnectedLayer.h"
#include "utils/MatrixUtils.h"

FullyConnectedLayer::FullyConnectedLayer(int outputSize, ActivationFunction* activationFunction)
    : outputSize(outputSize), activationFunction(activationFunction) {}

void FullyConnectedLayer::initialize(const std::vector<int>& inputShape) {
    if (inputShape.size() != 1) {
        throw std::invalid_argument("Expected input shape with 1 dimension (input size).");
    }
    inputSize = inputShape[0];
    weights.resize(inputSize, std::vector<double>(outputSize));
    biases.resize(outputSize, 0.0);
    initializeWeights();
    initializeAccumulatedGradients();
}

void FullyConnectedLayer::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std::sqrt(2.0 / inputSize));

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[i][j] = dist(gen);
        }
    }
}

void FullyConnectedLayer::initializeAccumulatedGradients() {
    accumulatedWeightGradients.resize(inputSize, std::vector<double>(outputSize, 0.0));
    accumulatedBiasGradients.resize(outputSize, 0.0);
}

std::vector<std::vector<std::vector<double>>> FullyConnectedLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    if (input[0][0].size() != inputSize) {
        throw std::invalid_argument("Input dimensions do not match the initialized shape.");
    }

    this->input = input;
    std::vector<double> flattenedInput = input[0][0];
    std::vector<double> preActivation = MatrixUtils::multiply(flattenedInput, weights, biases);
    std::vector<double> postActivation(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        postActivation[i] = activationFunction->activate(preActivation[i]);
    }

    return { { postActivation } };
}

std::vector<std::vector<std::vector<double>>> FullyConnectedLayer::backward(const std::vector<std::vector<std::vector<double>>>& gradient) {
    std::vector<double> postActivationGradient = gradient[0][0];
    std::vector<double> preActivationGradient(outputSize);

    std::vector<double> flattenedInput = input[0][0];
    std::vector<double> preActivation = MatrixUtils::multiply(flattenedInput, weights, biases);

    for (int i = 0; i < outputSize; ++i) {
        preActivationGradient[i] = postActivationGradient[i] * activationFunction->derivative(preActivation[i]);
    }

    std::vector<double> inputGradient(inputSize);
    std::vector<std::vector<double>> weightGradient(inputSize, std::vector<double>(outputSize));
    std::vector<double> biasGradient(outputSize);

    for (int j = 0; j < outputSize; ++j) {
        for (int i = 0; i < inputSize; ++i) {
            inputGradient[i] += preActivationGradient[j] * weights[i][j];
            weightGradient[i][j] += preActivationGradient[j] * flattenedInput[i];
        }
        biasGradient[j] += preActivationGradient[j];
    }

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            accumulatedWeightGradients[i][j] += weightGradient[i][j];
        }
    }
    for (int j = 0; j < outputSize; ++j) {
        accumulatedBiasGradients[j] += biasGradient[j];
    }

    return { { inputGradient } };
}

void FullyConnectedLayer::updateParameters(double learningRate, int miniBatchSize) {
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[i][j] -= learningRate * accumulatedWeightGradients[i][j] / miniBatchSize;
            accumulatedWeightGradients[i][j] = 0;
        }
    }
    for (int j = 0; j < outputSize; ++j) {
        biases[j] -= learningRate * accumulatedBiasGradients[j] / miniBatchSize;
        accumulatedBiasGradients[j] = 0;
    }
}

void FullyConnectedLayer::resetGradients() {
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            accumulatedWeightGradients[i][j] = 0;
        }
    }
    for (int j = 0; j < outputSize; ++j) {
        accumulatedBiasGradients[j] = 0;
    }
}

std::vector<int> FullyConnectedLayer::getOutputShape(const std::vector<int>& inputShape) {
    return { outputSize };
}
