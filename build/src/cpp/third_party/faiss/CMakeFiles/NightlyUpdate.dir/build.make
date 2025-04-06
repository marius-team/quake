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

# Utility rule file for NightlyUpdate.

# Include any custom commands dependencies for this target.
include src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/compiler_depend.make

# Include the progress variables for this target.
include src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/progress.make

src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss && /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/bin/ctest -D NightlyUpdate

NightlyUpdate: src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate
NightlyUpdate: src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/build.make
.PHONY : NightlyUpdate

# Rule to build all files generated by this target.
src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/build: NightlyUpdate
.PHONY : src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/build

src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/clean:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss && $(CMAKE_COMMAND) -P CMakeFiles/NightlyUpdate.dir/cmake_clean.cmake
.PHONY : src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/clean

src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/depend:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mandukhaialimaa/UWMadison/744/research/quake /Users/mandukhaialimaa/UWMadison/744/research/quake/src/cpp/third_party/faiss /Users/mandukhaialimaa/UWMadison/744/research/quake/build /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss /Users/mandukhaialimaa/UWMadison/744/research/quake/build/src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpp/third_party/faiss/CMakeFiles/NightlyUpdate.dir/depend

