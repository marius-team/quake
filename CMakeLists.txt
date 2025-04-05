cmake_minimum_required(VERSION 3.16.2)
project(quake_c VERSION 0.1 LANGUAGES CXX)

# ---------------------------------------------------------------
# Important System Variables and Their Defaults
# ---------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(Torch_USE_CUDA OFF CACHE BOOL "Force disable CUDA in Torch")
set(Torch_NO_CUDA ON CACHE BOOL "Force disable CUDA in Torch")
set(USE_CUDA OFF CACHE BOOL "Force disable CUDA globally")

# QUAKE_ENABLE_GPU: Enable GPU support for Faiss
# Default: OFF
if(QUAKE_ENABLE_GPU)
    set(FAISS_ENABLE_GPU ON)
else()
    set(FAISS_ENABLE_GPU OFF)
endif()

if(QUAKE_USE_NUMA)
    add_compile_definitions(QUAKE_USE_NUMA)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT DEFINED QUAKE_SET_ABI_MODE)
    set(QUAKE_SET_ABI_MODE ON)
endif()

# If in a conda environment, favor conda packages
if(EXISTS $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected. Using conda packages.")
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
endif()

# ---------------------------------------------------------------
# Find Torch Path via Python
# ---------------------------------------------------------------
execute_process(
        COMMAND python3 -c "import torch, os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath} ${CMAKE_PREFIX_PATH})

# ---------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------
set(CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(project_INCLUDE_DIR ${CPP_SOURCE}/include)
set(project_SOURCE_DIR ${CPP_SOURCE}/src)
set(project_BINDINGS_DIR ${CPP_SOURCE}/bindings)
set(project_THIRD_PARTY_DIR ${CPP_SOURCE}/third_party)
set(project_TEST_DIR test/cpp)

# ---------------------------------------------------------------
# Print out Compiler and Path Information
# ---------------------------------------------------------------
message(STATUS "--------- Build Configuration Summary ---------")
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Version: ${PROJECT_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "QUAKE_ENABLE_GPU: ${QUAKE_ENABLE_GPU} (FAISS_ENABLE_GPU: ${FAISS_ENABLE_GPU})")
message(STATUS "QUAKE_USE_NUMA: ${QUAKE_USE_NUMA}")

# Apple-specific adjustments
if(APPLE)
    include_directories("/opt/homebrew/opt/openblas/include")
    set(FAISS_ENABLE_GPU OFF)
endif()

# Compiler options and definitions
add_compile_options(-march=native)

# Switch ABI mode
if(QUAKE_SET_ABI_MODE)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()


# ---------------------------------------------------------------
# Find Required Packages
# ---------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(Arrow REQUIRED)  # Corrected to uppercase 'Arrow'
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)

message(STATUS "Torch include dir: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")

set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})

add_subdirectory(${project_THIRD_PARTY_DIR})

# ---------------------------------------------------------------
# Sources and Library
# ---------------------------------------------------------------
file(GLOB_RECURSE project_SOURCES ${project_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} SHARED ${project_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${project_INCLUDE_DIR}
        ${project_THIRD_PARTY_DIR}/concurrentqueue/
        ${ARROW_INCLUDE_DIR}
        faiss
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# ---------------------------------------------------------------
# Set Up Link Libraries Conditionally
# ---------------------------------------------------------------
# Base libraries common to all platforms
set(LINK_LIBS
        ${TORCH_LIBRARIES}
        -ffast-math
        -lpthread
        -fPIC
)

if(APPLE)
    list(APPEND LINK_LIBS faiss)
elseif(UNIX)
    # Linux or other UNIX-likes
    if (QUAKE_USE_AVX512)
        list(APPEND LINK_LIBS faiss_avx512)
    else()
        list(APPEND LINK_LIBS faiss)
    endif()

    if(QUAKE_USE_NUMA)
        list(APPEND LINK_LIBS -lnuma)
    endif()
else()
    # unsupported platform
    message(FATAL_ERROR "Unsupported platform")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${LINK_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC Arrow::arrow_shared)

IF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_TSAN)
    message("Using thread sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_TSAN)

IF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_ASAN)
    message("Using address sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND QUAKE_USE_ASAN)

# ---------------------------------------------------------------
# Python Bindings
# ---------------------------------------------------------------
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
file(GLOB_RECURSE BINDINGS ${project_BINDINGS_DIR}/*.cpp)
pybind11_add_module(_bindings ${BINDINGS})
set_target_properties(_bindings PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(_bindings PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
add_custom_target(bindings)
add_dependencies(bindings _bindings)

# ---------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------
file(GLOB BENCHMARK_SOURCES "src/cpp/test/benchmark/*.cpp")

foreach(SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE})
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${LINK_LIBS} ${PROJECT_NAME})
    target_include_directories(${EXECUTABLE_NAME}
            PUBLIC
            ${TORCH_INCLUDE_DIRS}
            ${project_INCLUDE_DIR}
            faiss
    )
    target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17)
endforeach()

# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(${project_TEST_DIR})
endif()

# ---------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------
message(STATUS "--------- Final Configuration Summary ---------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "GPU Enabled: ${FAISS_ENABLE_GPU}")
message(STATUS "NUMA Enabled: ${QUAKE_USE_NUMA}")
message(STATUS "Python used: ${Python3_EXECUTABLE}")
message(STATUS "Torch Path: ${TorchPath}")
message(STATUS "MKL_ARCH: ${MKL_ARCH}")
message(STATUS "MKL_ROOT: ${MKL_ROOT}")
message(STATUS "MKL_LINK: ${MKL_LINK}")
message(STATUS "MKL_INTERFACE_FULL: ${MKL_INTERFACE_FULL}")
message(STATUS "MKL_THREADING: ${MKL_THREADING}")
message(STATUS "MKL_MPI: ${MKL_MPI}")
message(STATUS "------------------------------------------------")