# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.12.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.12.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/clin99/cpp-taskflow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/clin99/cpp-taskflow/build

# Include any dependencies generated for this target.
include CMakeFiles/subflow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/subflow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/subflow.dir/flags.make

CMakeFiles/subflow.dir/example/subflow.cpp.o: CMakeFiles/subflow.dir/flags.make
CMakeFiles/subflow.dir/example/subflow.cpp.o: ../example/subflow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/subflow.dir/example/subflow.cpp.o"
	/usr/local/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/subflow.dir/example/subflow.cpp.o -c /Users/clin99/cpp-taskflow/example/subflow.cpp

CMakeFiles/subflow.dir/example/subflow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/subflow.dir/example/subflow.cpp.i"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clin99/cpp-taskflow/example/subflow.cpp > CMakeFiles/subflow.dir/example/subflow.cpp.i

CMakeFiles/subflow.dir/example/subflow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/subflow.dir/example/subflow.cpp.s"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clin99/cpp-taskflow/example/subflow.cpp -o CMakeFiles/subflow.dir/example/subflow.cpp.s

# Object files for target subflow
subflow_OBJECTS = \
"CMakeFiles/subflow.dir/example/subflow.cpp.o"

# External object files for target subflow
subflow_EXTERNAL_OBJECTS =

../example/subflow: CMakeFiles/subflow.dir/example/subflow.cpp.o
../example/subflow: CMakeFiles/subflow.dir/build.make
../example/subflow: CMakeFiles/subflow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../example/subflow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/subflow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/subflow.dir/build: ../example/subflow

.PHONY : CMakeFiles/subflow.dir/build

CMakeFiles/subflow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/subflow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/subflow.dir/clean

CMakeFiles/subflow.dir/depend:
	cd /Users/clin99/cpp-taskflow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build/CMakeFiles/subflow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/subflow.dir/depend

