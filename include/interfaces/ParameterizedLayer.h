#ifndef PARAMETERIZED_LAYER_H
#define PARAMETERIZED_LAYER_H

#include "interfaces/Layer.h"

class ParameterizedLayer : public virtual Layer {
public:
    virtual ~ParameterizedLayer() = default;

    // Updates the parameters of the layer using the accumulated gradients.
    virtual void updateParameters(double learningRate, int miniBatchSize) = 0;

    // Resets the accumulated gradients to zero.
    virtual void resetGradients() = 0;
};

#endif // PARAMETERIZED_LAYER_H