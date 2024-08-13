#include "utils/activationFunctions/ELU.h"
#include <cmath>

ELU::ELU(double alpha) : alpha(alpha) {}

double ELU::activate(double x) const {
    return x > 0 ? x : alpha * (std::exp(x) - 1);
}

double ELU::derivative(double x) const {
    return x > 0 ? 1 : alpha * std::exp(x);
}