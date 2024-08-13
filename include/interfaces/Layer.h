#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    virtual ~Layer() = default;
    
    virtual std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) = 0;
    
    virtual std::vector<std::vector<std::vector<double>>> backward(std::vector<std::vector<std::vector<double>>> gradient) = 0;
    
    virtual std::vector<int> getOutputShape(const std::vector<int>& inputShape) = 0;
};

#endif // LAYER_H