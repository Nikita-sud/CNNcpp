#ifndef CNN_H
#define CNN_H

#include "Layer.h"
#include "AdaptiveLayer.h"
#include "ParameterizedLayer.h"
#include "ImageData.h"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

class CNN {
public:
    CNN(double learningRate, const std::vector<int>& inputShape);

    void addLayer(std::shared_ptr<Layer> layer);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& gradient);
    void updateParameters(int miniBatchSize);
    void resetGradients();
    void SGD(const std::vector<ImageData>& trainingData, int epochs, int miniBatchSize, const std::vector<ImageData>& testData, const std::string& saveFilePath);
    void SGD(const std::vector<ImageData>& trainingData, int epochs, int miniBatchSize, const std::vector<ImageData>& testData);
    int evaluate(const std::vector<ImageData>& testData);
    void printNetworkSummary() const;
    void saveNetwork(const std::string& filePath) const;
    static CNN loadNetwork(const std::string& filePath);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    double learningRate;
    std::vector<int> inputShape;
    std::vector<std::vector<int>> layerShapes;

    std::vector<std::vector<ImageData>> createMiniBatches(const std::vector<ImageData>& trainingData, int miniBatchSize);
    void updateMiniBatch(const std::vector<ImageData>& miniBatch, int miniBatchSize);
    std::vector<std::vector<std::vector<double>>> computeLossGradient(const std::vector<double>& output, const std::vector<double>& target);
    int argMax(const std::vector<double>& array) const;
};

#endif // CNN_H