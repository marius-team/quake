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

# Include any dependencies generated for this target.
include test/cpp/CMakeFiles/quake_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include test/cpp/CMakeFiles/quake_tests.dir/progress.make

# Include the compile flags for this target's objects.
include test/cpp/CMakeFiles/quake_tests.dir/flags.make

test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o: ../test/cpp/benchmark.cpp
test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o -MF CMakeFiles/quake_tests.dir/benchmark.cpp.o.d -o CMakeFiles/quake_tests.dir/benchmark.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/benchmark.cpp

test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/benchmark.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/benchmark.cpp > CMakeFiles/quake_tests.dir/benchmark.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/benchmark.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/benchmark.cpp -o CMakeFiles/quake_tests.dir/benchmark.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o: ../test/cpp/dynamic_inverted_list.cpp
test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o -MF CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o.d -o CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/dynamic_inverted_list.cpp

test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/dynamic_inverted_list.cpp > CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/dynamic_inverted_list.cpp -o CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o: ../test/cpp/index_partition.cpp
test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o -MF CMakeFiles/quake_tests.dir/index_partition.cpp.o.d -o CMakeFiles/quake_tests.dir/index_partition.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/index_partition.cpp

test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/index_partition.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/index_partition.cpp > CMakeFiles/quake_tests.dir/index_partition.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/index_partition.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/index_partition.cpp -o CMakeFiles/quake_tests.dir/index_partition.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o: ../test/cpp/latency_estimator.cpp
test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o -MF CMakeFiles/quake_tests.dir/latency_estimator.cpp.o.d -o CMakeFiles/quake_tests.dir/latency_estimator.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/latency_estimator.cpp

test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/latency_estimator.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/latency_estimator.cpp > CMakeFiles/quake_tests.dir/latency_estimator.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/latency_estimator.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/latency_estimator.cpp -o CMakeFiles/quake_tests.dir/latency_estimator.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o: ../test/cpp/list_scanning.cpp
test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o -MF CMakeFiles/quake_tests.dir/list_scanning.cpp.o.d -o CMakeFiles/quake_tests.dir/list_scanning.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/list_scanning.cpp

test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/list_scanning.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/list_scanning.cpp > CMakeFiles/quake_tests.dir/list_scanning.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/list_scanning.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/list_scanning.cpp -o CMakeFiles/quake_tests.dir/list_scanning.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o: ../test/cpp/main.cpp
test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o -MF CMakeFiles/quake_tests.dir/main.cpp.o.d -o CMakeFiles/quake_tests.dir/main.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/main.cpp

test/cpp/CMakeFiles/quake_tests.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/main.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/main.cpp > CMakeFiles/quake_tests.dir/main.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/main.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/main.cpp -o CMakeFiles/quake_tests.dir/main.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o: ../test/cpp/maintenance.cpp
test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o -MF CMakeFiles/quake_tests.dir/maintenance.cpp.o.d -o CMakeFiles/quake_tests.dir/maintenance.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/maintenance.cpp

test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/maintenance.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/maintenance.cpp > CMakeFiles/quake_tests.dir/maintenance.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/maintenance.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/maintenance.cpp -o CMakeFiles/quake_tests.dir/maintenance.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o: ../test/cpp/partition_manager.cpp
test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o -MF CMakeFiles/quake_tests.dir/partition_manager.cpp.o.d -o CMakeFiles/quake_tests.dir/partition_manager.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/partition_manager.cpp

test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/partition_manager.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/partition_manager.cpp > CMakeFiles/quake_tests.dir/partition_manager.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/partition_manager.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/partition_manager.cpp -o CMakeFiles/quake_tests.dir/partition_manager.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o: ../test/cpp/quake_index.cpp
test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o -MF CMakeFiles/quake_tests.dir/quake_index.cpp.o.d -o CMakeFiles/quake_tests.dir/quake_index.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/quake_index.cpp

test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/quake_index.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/quake_index.cpp > CMakeFiles/quake_tests.dir/quake_index.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/quake_index.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/quake_index.cpp -o CMakeFiles/quake_tests.dir/quake_index.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o: ../test/cpp/query_coordinator.cpp
test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o -MF CMakeFiles/quake_tests.dir/query_coordinator.cpp.o.d -o CMakeFiles/quake_tests.dir/query_coordinator.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/query_coordinator.cpp

test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/query_coordinator.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/query_coordinator.cpp > CMakeFiles/quake_tests.dir/query_coordinator.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/query_coordinator.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/query_coordinator.cpp -o CMakeFiles/quake_tests.dir/query_coordinator.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o: ../test/cpp/search_recall_tests.cpp
test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o -MF CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o.d -o CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/search_recall_tests.cpp

test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/search_recall_tests.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/search_recall_tests.cpp > CMakeFiles/quake_tests.dir/search_recall_tests.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/search_recall_tests.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/search_recall_tests.cpp -o CMakeFiles/quake_tests.dir/search_recall_tests.cpp.s

test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/flags.make
test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o: ../test/cpp/topk_buffer.cpp
test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o: test/cpp/CMakeFiles/quake_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o -MF CMakeFiles/quake_tests.dir/topk_buffer.cpp.o.d -o CMakeFiles/quake_tests.dir/topk_buffer.cpp.o -c /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/topk_buffer.cpp

test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quake_tests.dir/topk_buffer.cpp.i"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/topk_buffer.cpp > CMakeFiles/quake_tests.dir/topk_buffer.cpp.i

test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quake_tests.dir/topk_buffer.cpp.s"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp/topk_buffer.cpp -o CMakeFiles/quake_tests.dir/topk_buffer.cpp.s

# Object files for target quake_tests
quake_tests_OBJECTS = \
"CMakeFiles/quake_tests.dir/benchmark.cpp.o" \
"CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o" \
"CMakeFiles/quake_tests.dir/index_partition.cpp.o" \
"CMakeFiles/quake_tests.dir/latency_estimator.cpp.o" \
"CMakeFiles/quake_tests.dir/list_scanning.cpp.o" \
"CMakeFiles/quake_tests.dir/main.cpp.o" \
"CMakeFiles/quake_tests.dir/maintenance.cpp.o" \
"CMakeFiles/quake_tests.dir/partition_manager.cpp.o" \
"CMakeFiles/quake_tests.dir/quake_index.cpp.o" \
"CMakeFiles/quake_tests.dir/query_coordinator.cpp.o" \
"CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o" \
"CMakeFiles/quake_tests.dir/topk_buffer.cpp.o"

# External object files for target quake_tests
quake_tests_EXTERNAL_OBJECTS =

test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/benchmark.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/dynamic_inverted_list.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/index_partition.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/latency_estimator.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/list_scanning.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/main.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/maintenance.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/partition_manager.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/quake_index.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/query_coordinator.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/search_recall_tests.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/topk_buffer.cpp.o
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/build.make
test/cpp/quake_tests: libquake_c.dylib
test/cpp/quake_tests: lib/libgtest.a
test/cpp/quake_tests: lib/libgtest_main.a
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/python3.11/site-packages/torch/lib/libtorch.dylib
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/python3.11/site-packages/torch/lib/libtorch_cpu.dylib
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/python3.11/site-packages/torch/lib/libc10.dylib
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/python3.11/site-packages/torch/lib/libc10.dylib
test/cpp/quake_tests: src/cpp/third_party/faiss/faiss/libfaiss.a
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/libomp.dylib
test/cpp/quake_tests: /opt/homebrew/Caskroom/miniconda/base/envs/quake-env/lib/libopenblas.dylib
test/cpp/quake_tests: lib/libgtest.a
test/cpp/quake_tests: test/cpp/CMakeFiles/quake_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/mandukhaialimaa/UWMadison/744/research/quake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX executable quake_tests"
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quake_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/cpp/CMakeFiles/quake_tests.dir/build: test/cpp/quake_tests
.PHONY : test/cpp/CMakeFiles/quake_tests.dir/build

test/cpp/CMakeFiles/quake_tests.dir/clean:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp && $(CMAKE_COMMAND) -P CMakeFiles/quake_tests.dir/cmake_clean.cmake
.PHONY : test/cpp/CMakeFiles/quake_tests.dir/clean

test/cpp/CMakeFiles/quake_tests.dir/depend:
	cd /Users/mandukhaialimaa/UWMadison/744/research/quake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mandukhaialimaa/UWMadison/744/research/quake /Users/mandukhaialimaa/UWMadison/744/research/quake/test/cpp /Users/mandukhaialimaa/UWMadison/744/research/quake/build /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp /Users/mandukhaialimaa/UWMadison/744/research/quake/build/test/cpp/CMakeFiles/quake_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/cpp/CMakeFiles/quake_tests.dir/depend

