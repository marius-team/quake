# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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
CMAKE_COMMAND = /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mandukhaialimaa/UWMadison/744/research/quake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mandukhaialimaa/UWMadison/744/research/quake/build

# Utility rule file for ExperimentalBuild.

# Include any custom commands dependencies for this target.
include src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/compiler_depend.make

# Include the progress variables for this target.
include src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/progress.make

src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss && /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/bin/ctest -D ExperimentalBuild

ExperimentalBuild: src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild
ExperimentalBuild: src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/build.make
.PHONY : ExperimentalBuild

# Rule to build all files generated by this target.
src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/build: ExperimentalBuild
.PHONY : src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/build

src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/clean:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalBuild.dir/cmake_clean.cmake
.PHONY : src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/clean

src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/depend:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mandukhaialimaa/UWMadison/744/research/quake /Users/mandukhaialimaa/UWMadison/744/research/quake/src/cpp/third_party/faiss /Users/mandukhaialimaa/UWMadison/744/research/quake/build /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpp/third_party/faiss/CMakeFiles/ExperimentalBuild.dir/depend

