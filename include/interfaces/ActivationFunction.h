#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    // Applies the activation function to a single input value.
    virtual double activate(double x) const = 0;

    // Computes the derivative of the activation function for a given input value.
    virtual double derivative(double x) const = 0;

    // Applies the activation function to an array of input values.
    virtual std::vector<double> activate(const std::vector<double>& input) const {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = activate(input[i]);
        }
        return output;
    }
};

#endif // ACTIVATION_FUNCTION_H