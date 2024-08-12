#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>

class MatrixUtils {
public:
    static double applyFilter(const std::vector<std::vector<double>>& input, 
                              const std::vector<std::vector<double>>& filter, 
                              int startX, int startY);

    static std::vector<std::vector<double>> rotate180(const std::vector<std::vector<double>>& matrix);

    static std::vector<std::vector<double>> convolve(const std::vector<std::vector<double>>& input, 
                                                     const std::vector<std::vector<double>>& filter, 
                                                     int stride);

    static std::vector<std::vector<double>> fullConvolve(const std::vector<std::vector<double>>& input, 
                                                         const std::vector<std::vector<double>>& filter);

    static std::vector<std::vector<double>> maxPooling(const std::vector<std::vector<double>>& input, 
                                                       int poolSize);

    static std::vector<std::vector<double>> averagePooling(const std::vector<std::vector<double>>& input, 
                                                           int poolSize);

    static std::vector<double> multiply(const std::vector<double>& input, 
                                        const std::vector<std::vector<double>>& weights, 
                                        const std::vector<double>& biases);

    static std::vector<std::vector<std::vector<double>>> add(const std::vector<std::vector<std::vector<double>>>& a, 
                                                             const std::vector<std::vector<std::vector<double>>>& b);

    static std::vector<std::vector<std::vector<double>>> divide(const std::vector<std::vector<std::vector<double>>>& a, 
                                                                double scalar);

    static std::vector<std::vector<std::vector<double>>> unflatten(const std::vector<double>& input, 
                                                                   int depth, int height, int width);
};

#endif // MATRIX_UTILS_H