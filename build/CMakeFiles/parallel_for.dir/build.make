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
include CMakeFiles/parallel_for.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/parallel_for.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parallel_for.dir/flags.make

CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o: CMakeFiles/parallel_for.dir/flags.make
CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o: ../example/parallel_for.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o"
	/usr/local/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o -c /Users/clin99/cpp-taskflow/example/parallel_for.cpp

CMakeFiles/parallel_for.dir/example/parallel_for.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parallel_for.dir/example/parallel_for.cpp.i"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/clin99/cpp-taskflow/example/parallel_for.cpp > CMakeFiles/parallel_for.dir/example/parallel_for.cpp.i

CMakeFiles/parallel_for.dir/example/parallel_for.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parallel_for.dir/example/parallel_for.cpp.s"
	/usr/local/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/clin99/cpp-taskflow/example/parallel_for.cpp -o CMakeFiles/parallel_for.dir/example/parallel_for.cpp.s

# Object files for target parallel_for
parallel_for_OBJECTS = \
"CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o"

# External object files for target parallel_for
parallel_for_EXTERNAL_OBJECTS =

../example/parallel_for: CMakeFiles/parallel_for.dir/example/parallel_for.cpp.o
../example/parallel_for: CMakeFiles/parallel_for.dir/build.make
../example/parallel_for: CMakeFiles/parallel_for.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/clin99/cpp-taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../example/parallel_for"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parallel_for.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parallel_for.dir/build: ../example/parallel_for

.PHONY : CMakeFiles/parallel_for.dir/build

CMakeFiles/parallel_for.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parallel_for.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parallel_for.dir/clean

CMakeFiles/parallel_for.dir/depend:
	cd /Users/clin99/cpp-taskflow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build /Users/clin99/cpp-taskflow/build/CMakeFiles/parallel_for.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parallel_for.dir/depend

