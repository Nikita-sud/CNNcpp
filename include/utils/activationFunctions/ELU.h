#ifndef ELU_H
#define ELU_H

#include "interfaces/ActivationFunction.h"

class ELU : public ActivationFunction {
public:
    explicit ELU(double alpha);

    double activate(double x) const override;
    double derivative(double x) const override;

private:
    double alpha;
};

#endif // ELU_H