# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-src"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-build"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/mandukhaialimaa/UWMadison/744/research/quake/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
