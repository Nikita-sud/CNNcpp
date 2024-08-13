cmake_minimum_required(VERSION 3.10)
project(CNNcpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the sources
set(SOURCES
    src/main.cpp
    src/cnn/CNN.cpp
    src/cnn/MNISTReader.cpp
    src/layers/ConvolutionalLayer.cpp
    src/layers/FlattenLayer.cpp
    src/layers/FullyConnectedLayer.cpp
    src/layers/SoftmaxLayer.cpp
    src/utils/MatrixUtils.cpp
    src/utils/ImageData.cpp
    src/utils/activationFunctions/ReLU.cpp
    src/utils/activationFunctions/ELU.cpp  
    # Add other source files here
)

# Define the include directories
include_directories(include)

# Add executable
add_executable(CNNcpp ${SOURCES})

# Link Metal framework
find_library(METAL Metal)
find_library(METALKIT MetalKit)

target_link_libraries(CNNcpp ${METAL} ${METALKIT})

# If you have any Metal shader sources, you can add them using:
# set_source_files_properties(shaders/your_shader.metal PROPERTIES LANGUAGE METAL)