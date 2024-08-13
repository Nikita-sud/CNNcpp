#include "cnn/CNN.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/SoftmaxLayer.h"
#include "utils/activationFunctions/ELU.h"
#include "cnn/MNISTReader.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

int main() {
    double learningRate = 0.02;
    CNN cnn(learningRate, {1, 28, 28});

    cnn.addLayer(std::make_shared<FlattenLayer>());
    cnn.addLayer(std::make_shared<FullyConnectedLayer>(60, std::make_shared<ELU>(1.0)));
    cnn.addLayer(std::make_shared<FullyConnectedLayer>(10, std::make_shared<ELU>(1.0)));
    cnn.addLayer(std::make_shared<SoftmaxLayer>());

    cnn.printNetworkSummary();

    std::string trainImagesFile = "../data/train-images.idx3-ubyte";
    std::string trainLabelsFile = "../data/train-labels.idx1-ubyte";
    auto trainDataset = MNISTReader::readMNISTData(trainImagesFile, trainLabelsFile);

    std::string testImagesFile = "../data/t10k-images.idx3-ubyte";
    std::string testLabelsFile = "../data/t10k-labels.idx1-ubyte";
    auto testDataset = MNISTReader::readMNISTData(testImagesFile, testLabelsFile);

    cnn.SGD(trainDataset, 30, 32, testDataset);

    auto input = testDataset[2].getImageData();
    auto output = cnn.forward(input);

    std::cout << "CNN output: " << output[0][0] << std::endl;
    std::cout << "Actual output: " << testDataset[2].getLabel() << std::endl;

    return 0;
}