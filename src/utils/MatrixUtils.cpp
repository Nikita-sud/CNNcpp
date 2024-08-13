#include "utils/MatrixUtils.h"
#include <algorithm>
#include <cmath>

double MatrixUtils::applyFilter(const std::vector<std::vector<double>>& input, 
                                const std::vector<std::vector<double>>& filter, 
                                int startX, int startY) {
    int filterSize = filter.size();
    double sum = 0;

    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            int x = startX + i;
            int y = startY + j;
            if (x >= 0 && x < input.size() && y >= 0 && y < input[0].size()) {
                sum += input[x][y] * filter[i][j];
            }
        }
    }
    return sum;
}

std::vector<std::vector<double>> MatrixUtils::rotate180(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();
    std::vector<std::vector<double>> rotated(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            rotated[i][j] = matrix[n - 1 - i][n - 1 - j];
        }
    }
    return rotated;
}

std::vector<std::vector<double>> MatrixUtils::convolve(const std::vector<std::vector<double>>& input, 
                                                       const std::vector<std::vector<double>>& filter, 
                                                       int stride) {
    int inputSize = input.size();
    int filterSize = filter.size();
    int outputSize = (inputSize - filterSize) / stride + 1;
    std::vector<std::vector<double>> output(outputSize, std::vector<double>(outputSize));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            output[i][j] = applyFilter(input, filter, i * stride, j * stride);
        }
    }
    return output;
}

std::vector<std::vector<double>> MatrixUtils::fullConvolve(const std::vector<std::vector<double>>& input, 
                                                           const std::vector<std::vector<double>>& filter) {
    int inputSize = input.size();
    int filterSize = filter.size();
    int outputSize = inputSize + filterSize - 1;
    std::vector<std::vector<double>> output(outputSize, std::vector<double>(outputSize));

    for (int i = -filterSize + 1; i < inputSize; ++i) {
        for (int j = -filterSize + 1; j < inputSize; ++j) {
            output[i + filterSize - 1][j + filterSize - 1] = applyFilter(input, filter, i, j);
        }
    }
    return output;
}

std::vector<std::vector<double>> MatrixUtils::maxPooling(const std::vector<std::vector<double>>& input, 
                                                         int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;
    std::vector<std::vector<double>> output(outputSize, std::vector<double>(outputSize));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            double maxVal = input[i * poolSize][j * poolSize];
            for (int k = 0; k < poolSize; ++k) {
                for (int l = 0; l < poolSize; ++l) {
                    maxVal = std::max(maxVal, input[i * poolSize + k][j * poolSize + l]);
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
}

std::vector<std::vector<double>> MatrixUtils::averagePooling(const std::vector<std::vector<double>>& input, 
                                                             int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;
    std::vector<std::vector<double>> output(outputSize, std::vector<double>(outputSize));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            double sum = 0.0;
            for (int k = 0; k < poolSize; ++k) {
                for (int l = 0; l < poolSize; ++l) {
                    sum += input[i * poolSize + k][j * poolSize + l];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
        }
    }
    return output;
}

std::vector<double> MatrixUtils::multiply(const std::vector<double>& input, 
                                          const std::vector<std::vector<double>>& weights, 
                                          const std::vector<double>& biases) {
    int inputSize = input.size();
    int outputSize = biases.size();
    std::vector<double> output(outputSize);

    for (int j = 0; j < outputSize; ++j) {
        double sum = biases[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[i][j];
        }
        output[j] = sum;
    }
    return output;
}

std::vector<std::vector<std::vector<double>>> MatrixUtils::add(const std::vector<std::vector<std::vector<double>>>& a, 
                                                               const std::vector<std::vector<std::vector<double>>>& b) {
    int depth = a.size();
    int height = a[0].size();
    int width = a[0][0].size();
    std::vector<std::vector<std::vector<double>>> result(depth, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                result[d][h][w] = a[d][h][w] + b[d][h][w];
            }
        }
    }
    return result;
}

std::vector<std::vector<std::vector<double>>> MatrixUtils::divide(const std::vector<std::vector<std::vector<double>>>& a, 
                                                                  double scalar) {
    int depth = a.size();
    int height = a[0].size();
    int width = a[0][0].size();
    std::vector<std::vector<std::vector<double>>> result(depth, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                result[d][h][w] = a[d][h][w] / scalar;
            }
        }
    }
    return result;
}

std::vector<std::vector<std::vector<double>>> MatrixUtils::unflatten(const std::vector<double>& input, 
                                                                     int depth, int height, int width) {
    std::vector<std::vector<std::vector<double>>> unflattened(depth, std::vector<std::vector<double>>(height, std::vector<double>(width)));
    
    int index = 0;
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                unflattened[d][h][w] = input[index++];
            }
        }
    }
    return unflattened;
}
