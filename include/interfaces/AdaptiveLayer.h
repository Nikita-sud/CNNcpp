#ifndef ADAPTIVE_LAYER_H
#define ADAPTIVE_LAYER_H

#include "interfaces/Layer.h"
#include <vector>
#include <stdexcept>

class AdaptiveLayer : public virtual Layer {
public:
    virtual ~AdaptiveLayer() = default;

    // Initializes the layer with the given input shape.
    virtual void initialize(const std::vector<int>& inputShape) = 0;
};

#endif // ADAPTIVE_LAYER_H