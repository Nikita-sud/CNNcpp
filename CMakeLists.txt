# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(CNNProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Define the sources
set(SOURCES
    src/main.cpp
    src/layers/ConvolutionalLayer.cpp
    src/interfaces/Layer.cpp
    # Add other source files here
)

# Define the include directories
include_directories(include)

# Add executable
add_executable(CNNProject ${SOURCES})

# Link libraries (e.g., OpenGL, GLEW, etc.)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
target_link_libraries(CNNProject ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})