#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    virtual ~Layer() = default;

    // Performs the forward pass through the layer.
    virtual std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input) = 0;

    // Performs the backward pass through the layer.
    virtual std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>>& gradient) = 0;

    // Computes the output shape of the layer given the input shape.
    virtual std::vector<int> getOutputShape(const std::vector<int>& inputShape) = 0;
};

#endif // LAYER_H