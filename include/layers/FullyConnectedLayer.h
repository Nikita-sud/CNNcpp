#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "AdaptiveLayer.h"
#include "ParameterizedLayer.h"
#include "ActivationFunction.h"
#include <vector>
#include <random>

class FullyConnectedLayer : public AdaptiveLayer, public ParameterizedLayer {
public:
    FullyConnectedLayer(int outputSize, ActivationFunction* activationFunction);

    void initialize(const std::vector<int>& inputShape) override;
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) override;
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& gradient) override;
    void updateParameters(double learningRate, int miniBatchSize) override;
    void resetGradients() override;
    std::vector<int> getOutputShape(const std::vector<int>& inputShape) override;

private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<std::vector<std::vector<double>>> input;
    ActivationFunction* activationFunction;
    std::vector<std::vector<double>> accumulatedWeightGradients;
    std::vector<double> accumulatedBiasGradients;

    void initializeWeights();
    void initializeAccumulatedGradients();
};

#endif // FULLY_CONNECTED_LAYER_H
