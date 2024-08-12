#include "ReLU.h"
#include <algorithm>

double ReLU::activate(double x) const {
    return std::max(0.0, x);
}

double ReLU::derivative(double x) const {
    return x > 0 ? 1.0 : 0.0;
}