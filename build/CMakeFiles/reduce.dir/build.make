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
include CMakeFiles/reduce.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduce.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduce.dir/flags.make

CMakeFiles/reduce.dir/example/reduce.cpp.o: CMakeFiles/reduce.dir/flags.make
CMakeFiles/reduce.dir/example/reduce.cpp.o: ../example/reduce.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reduce.dir/example/reduce.cpp.o"
	/usr/local/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reduce.dir/example/reduce.cpp.o -c /Users/clin99/cpp-taskflow/example/reduce.cpp

CMakeFiles/reduce.dir/example/reduce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reduce.dir/example/reduce.cpp.i"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clin99/cpp-taskflow/example/reduce.cpp > CMakeFiles/reduce.dir/example/reduce.cpp.i

CMakeFiles/reduce.dir/example/reduce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reduce.dir/example/reduce.cpp.s"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clin99/cpp-taskflow/example/reduce.cpp -o CMakeFiles/reduce.dir/example/reduce.cpp.s

# Object files for target reduce
reduce_OBJECTS = \
"CMakeFiles/reduce.dir/example/reduce.cpp.o"

# External object files for target reduce
reduce_EXTERNAL_OBJECTS =

../example/reduce: CMakeFiles/reduce.dir/example/reduce.cpp.o
../example/reduce: CMakeFiles/reduce.dir/build.make
../example/reduce: CMakeFiles/reduce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../example/reduce"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduce.dir/build: ../example/reduce

.PHONY : CMakeFiles/reduce.dir/build

CMakeFiles/reduce.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduce.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduce.dir/clean

CMakeFiles/reduce.dir/depend:
	cd /Users/clin99/cpp-taskflow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build/CMakeFiles/reduce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduce.dir/depend
