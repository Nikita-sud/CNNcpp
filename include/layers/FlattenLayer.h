#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "interfaces/AdaptiveLayer.h"
#include <vector>

class FlattenLayer : public AdaptiveLayer {
public:
    FlattenLayer();

    void initialize(const std::vector<int>& inputShape) override;
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) override;
    std::vector<std::vector<std::vector<double>>> backward(std::vector<std::vector<std::vector<double>>> gradient) override;
    std::vector<int> getOutputShape(const std::vector<int>& inputShape) override;

private:
    int depth;
    int height;
    int width;
};

#endif // FLATTEN_LAYER_H