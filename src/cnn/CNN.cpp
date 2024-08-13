#include "cnn/CNN.h"
#include <random> 

CNN::CNN(double learningRate, const std::vector<int>& inputShape)
    : learningRate(learningRate), inputShape(inputShape) {}

void CNN::addLayer(std::shared_ptr<Layer> layer) {
    std::vector<int> currentShape = inputShape;

    if (auto adaptiveLayer = std::dynamic_pointer_cast<AdaptiveLayer>(layer)) {
        adaptiveLayer->initialize(inputShape);
    }
    inputShape = layer->getOutputShape(currentShape);
    layers.push_back(layer);
    layerShapes.push_back(currentShape);
    layerShapes.push_back(inputShape);
}

std::vector<std::vector<std::vector<double>>> CNN::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    auto output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<std::vector<std::vector<double>>> CNN::backward(const std::vector<std::vector<std::vector<double>>>& gradient) {
    auto grad = gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void CNN::updateParameters(int miniBatchSize) {
    for (const auto& layer : layers) {
        if (auto paramLayer = std::dynamic_pointer_cast<ParameterizedLayer>(layer)) {
            paramLayer->updateParameters(learningRate, miniBatchSize);
        }
    }
}

void CNN::resetGradients() {
    for (const auto& layer : layers) {
        if (auto paramLayer = std::dynamic_pointer_cast<ParameterizedLayer>(layer)) {
            paramLayer->resetGradients();
        }
    }
}

void CNN::SGD(const std::vector<ImageData>& trainingData, int epochs, int miniBatchSize, const std::vector<ImageData>& testData, const std::string& saveFilePath) {
    int nTest = static_cast<int>(testData.size());
    double bestAccuracy = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto shuffledData = trainingData;
        
        // Use std::shuffle instead of std::random_shuffle
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffledData.begin(), shuffledData.end(), g);
        
        auto miniBatches = createMiniBatches(shuffledData, miniBatchSize);

        for (const auto& miniBatch : miniBatches) {
            updateMiniBatch(miniBatch, miniBatchSize);
        }

        if (nTest > 0) {
            int correct = evaluate(testData);
            double accuracy = static_cast<double>(correct) / nTest;
            std::cout << "Epoch " << (epoch + 1) << ": " << correct << " / " << nTest << " (" << accuracy * 100 << "%)\n";

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                saveNetwork(saveFilePath);
                std::cout << "New best model saved with accuracy: " << bestAccuracy * 100 << "%\n";
            }
        }
    }
}

void CNN::SGD(const std::vector<ImageData>& trainingData, int epochs, int miniBatchSize, const std::vector<ImageData>& testData) {
    int nTest = static_cast<int>(testData.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto shuffledData = trainingData;
        
        // Use std::shuffle instead of std::random_shuffle
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffledData.begin(), shuffledData.end(), g);
        
        auto miniBatches = createMiniBatches(shuffledData, miniBatchSize);

        for (const auto& miniBatch : miniBatches) {
            updateMiniBatch(miniBatch, miniBatchSize);
        }

        if (nTest > 0) {
            int correct = evaluate(testData);
            double accuracy = static_cast<double>(correct) / nTest;
            std::cout << "Epoch " << (epoch + 1) << ": " << correct << " / " << nTest << " (" << accuracy * 100 << "%)\n";
        }
    }
}

std::vector<std::vector<ImageData>> CNN::createMiniBatches(const std::vector<ImageData>& trainingData, int miniBatchSize) {
    std::vector<std::vector<ImageData>> miniBatches;
    for (size_t i = 0; i < trainingData.size(); i += miniBatchSize) {
        miniBatches.emplace_back(trainingData.begin() + i, trainingData.begin() + std::min(i + miniBatchSize, trainingData.size()));
    }
    return miniBatches;
}

void CNN::updateMiniBatch(const std::vector<ImageData>& miniBatch, int miniBatchSize) {
    resetGradients();
    for (const auto& data : miniBatch) {
        auto output = forward(data.getImageData());
        auto lossGradient = computeLossGradient(output[0][0], data.getLabel());
        backward(lossGradient);
    }
    updateParameters(miniBatchSize);
}

std::vector<std::vector<std::vector<double>>> CNN::computeLossGradient(const std::vector<double>& output, const std::vector<double>& target) {
    std::vector<std::vector<std::vector<double>>> gradient(1, std::vector<std::vector<double>>(1, std::vector<double>(output.size())));
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[0][0][i] = output[i] - target[i];
    }
    return gradient;
}

int CNN::evaluate(const std::vector<ImageData>& testData) {
    int correct = 0;
    for (const auto& data : testData) {
        auto output = forward(data.getImageData());
        int predictedLabel = argMax(output[0][0]);
        int actualLabel = argMax(data.getLabel());
        if (predictedLabel == actualLabel) {
            ++correct;
        }
    }
    return correct;
}

int CNN::argMax(const std::vector<double>& array) const {
    return static_cast<int>(std::distance(array.begin(), std::max_element(array.begin(), array.end())));
}

void CNN::printNetworkSummary() const {
    std::cout << "CNN Network Summary:\n";
    std::cout << "Number of layers: " << layers.size() << "\n";
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& inputShape = layerShapes[2 * i];
        const auto& outputShape = layerShapes[2 * i + 1];
        std::cout << "Layer " << (i + 1) << ": " << typeid(*layers[i]).name() << " -> Input Shape: [";
        for (const auto& dim : inputShape) std::cout << dim << " ";
        std::cout << "], Output Shape: [";
        for (const auto& dim : outputShape) std::cout << dim << " ";
        std::cout << "]\n";
    }
}

void CNN::saveNetwork(const std::string& filePath) const {
    std::ofstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        // Serialize the CNN object here (you would need to implement the actual serialization)
        file.close();
    } else {
        std::cerr << "Failed to open file for saving the network.\n";
    }
}

CNN CNN::loadNetwork(const std::string& filePath) {
    CNN loadedCNN(0.0, {});
    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        // Deserialize the CNN object here (you would need to implement the actual deserialization)
        file.close();
    } else {
        std::cerr << "Failed to open file for loading the network.\n";
    }
    return loadedCNN;
}
