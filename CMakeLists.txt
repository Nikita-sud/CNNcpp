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
    src/utils/MatrixUtils.cpp
    src/utils/ImageData.cpp        # Ensure this is added
    src/utils/activationFunctions/ReLU.cpp             # Ensure this is added
    # Add other source files here
)

# Define the include directories
include_directories(include)

# Add executable
add_executable(CNNcpp ${SOURCES})

# Link libraries (e.g., OpenGL, GLEW, GLUT)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# Link the OpenGL, GLEW, and GLUT libraries
target_link_libraries(CNNcpp OpenGL::GL GLEW::GLEW GLUT::GLUT)