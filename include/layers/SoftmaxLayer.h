#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "Layer.h"
#include <vector>

class SoftmaxLayer : public Layer {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) override;
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& gradient) override;
    std::vector<int> getOutputShape(const std::vector<int>& inputShape) override;

private:
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<double> softmax(const std::vector<double>& input);
};

#endif // SOFTMAX_LAYER_H
