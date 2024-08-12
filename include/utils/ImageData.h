#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <vector>

class ImageData {
public:
    ImageData(const std::vector<std::vector<std::vector<double>>>& imageData, const std::vector<double>& label);

    const std::vector<std::vector<std::vector<double>>>& getImageData() const;
    const std::vector<double>& getLabel() const;

private:
    std::vector<std::vector<std::vector<double>>> imageData;
    std::vector<double> label;
};

#endif // IMAGE_DATA_H