#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

class ReLU : public ActivationFunction {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

#endif // RELU_H