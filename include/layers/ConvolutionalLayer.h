#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "ActivationFunction.h"
#include "AdaptiveLayer.h"
#include "ParameterizedLayer.h"
#include <vector>
#include <random>

class ConvolutionalLayer : public AdaptiveLayer, public ParameterizedLayer {
public:
    ConvolutionalLayer(int filterSize, int numFilters, int stride, ActivationFunction* activationFunction);
    ConvolutionalLayer(int filterSize, int numFilters, ActivationFunction* activationFunction);
    ConvolutionalLayer(int filterSize, int numFilters);
    
    void initialize(const std::vector<int>& inputShape) override;
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) override;
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& gradient) override;
    void updateParameters(double learningRate, int miniBatchSize) override;
    void resetGradients() override;
    std::vector<int> getOutputShape(const std::vector<int>& inputShape) override;

private:
    int filterSize;
    int numFilters;
    int stride;
    std::vector<std::vector<std::vector<std::vector<double>>>> filters;
    std::vector<double> biases;
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<std::vector<std::vector<double>>> activatedOutput;
    ActivationFunction* activationFunction;
    std::vector<std::vector<std::vector<std::vector<double>>>> accumulatedFilterGradients;
    std::vector<double> accumulatedBiasGradients;

    void initializeFilters(int inputDepth);
    void initializeBiases();
    void initializeAccumulatedGradients();
};

#endif // CONVOLUTIONAL_LAYER_H