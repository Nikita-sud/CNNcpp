# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nichitabulgaru/Documents/CNNcpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nichitabulgaru/Documents/CNNcpp/build

# Include any dependencies generated for this target.
include CMakeFiles/CNNcpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CNNcpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CNNcpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CNNcpp.dir/flags.make

CMakeFiles/CNNcpp.dir/src/main.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/main.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/main.cpp
CMakeFiles/CNNcpp.dir/src/main.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CNNcpp.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/main.cpp.o -MF CMakeFiles/CNNcpp.dir/src/main.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/main.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/main.cpp

CMakeFiles/CNNcpp.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/main.cpp > CMakeFiles/CNNcpp.dir/src/main.cpp.i

CMakeFiles/CNNcpp.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/main.cpp -o CMakeFiles/CNNcpp.dir/src/main.cpp.s

CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/CNN.cpp
CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o -MF CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/CNN.cpp

CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/CNN.cpp > CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.i

CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/CNN.cpp -o CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.s

CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/MNISTReader.cpp
CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o -MF CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/MNISTReader.cpp

CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/MNISTReader.cpp > CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.i

CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/cnn/MNISTReader.cpp -o CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.s

CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/layers/ConvolutionalLayer.cpp
CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o -MF CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/layers/ConvolutionalLayer.cpp

CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/layers/ConvolutionalLayer.cpp > CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.i

CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/layers/ConvolutionalLayer.cpp -o CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.s

CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/utils/MatrixUtils.cpp
CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o -MF CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/utils/MatrixUtils.cpp

CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/utils/MatrixUtils.cpp > CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.i

CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/utils/MatrixUtils.cpp -o CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.s

CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/utils/ImageData.cpp
CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o -MF CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/utils/ImageData.cpp

CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/utils/ImageData.cpp > CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.i

CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/utils/ImageData.cpp -o CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.s

CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o: CMakeFiles/CNNcpp.dir/flags.make
CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o: /Users/nichitabulgaru/Documents/CNNcpp/src/utils/activationFunctions/ReLU.cpp
CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o: CMakeFiles/CNNcpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o -MF CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o.d -o CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o -c /Users/nichitabulgaru/Documents/CNNcpp/src/utils/activationFunctions/ReLU.cpp

CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nichitabulgaru/Documents/CNNcpp/src/utils/activationFunctions/ReLU.cpp > CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.i

CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nichitabulgaru/Documents/CNNcpp/src/utils/activationFunctions/ReLU.cpp -o CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.s

# Object files for target CNNcpp
CNNcpp_OBJECTS = \
"CMakeFiles/CNNcpp.dir/src/main.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o" \
"CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o"

# External object files for target CNNcpp
CNNcpp_EXTERNAL_OBJECTS =

CNNcpp: CMakeFiles/CNNcpp.dir/src/main.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/cnn/CNN.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/cnn/MNISTReader.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/layers/ConvolutionalLayer.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/utils/MatrixUtils.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/utils/ImageData.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/src/utils/activationFunctions/ReLU.cpp.o
CNNcpp: CMakeFiles/CNNcpp.dir/build.make
CNNcpp: /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/System/Library/Frameworks/OpenGL.framework
CNNcpp: /opt/homebrew/lib/libGLEW.2.2.0.dylib
CNNcpp: /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/System/Library/Frameworks/GLUT.framework
CNNcpp: /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/System/Library/Frameworks/Cocoa.framework
CNNcpp: CMakeFiles/CNNcpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable CNNcpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CNNcpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CNNcpp.dir/build: CNNcpp
.PHONY : CMakeFiles/CNNcpp.dir/build

CMakeFiles/CNNcpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CNNcpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CNNcpp.dir/clean

CMakeFiles/CNNcpp.dir/depend:
	cd /Users/nichitabulgaru/Documents/CNNcpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nichitabulgaru/Documents/CNNcpp /Users/nichitabulgaru/Documents/CNNcpp /Users/nichitabulgaru/Documents/CNNcpp/build /Users/nichitabulgaru/Documents/CNNcpp/build /Users/nichitabulgaru/Documents/CNNcpp/build/CMakeFiles/CNNcpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/CNNcpp.dir/depend

