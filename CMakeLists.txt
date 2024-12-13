cmake_minimum_required(VERSION 3.19.2)
project(quake_c VERSION 0.1 LANGUAGES CXX)

# Compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# favor conda packages over system packages
if (EXISTS $ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
endif ()

# set the TorchPath variable to the path of the torch library
execute_process(
        COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath} ${CMAKE_PREFIX_PATH})
set(CMAKE_MODULE_PATH ${project_CMAKE_DIR} ${CMAKE_MODULE_PATH})

set(CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(project_INCLUDE_DIR ${CPP_SOURCE}/include)
set(project_SOURCE_DIR ${CPP_SOURCE}/src)
set(project_BINDINGS_DIR ${CPP_SOURCE}/bindings)
set(project_THIRD_PARTY_DIR ${CPP_SOURCE}/third_party)
set(project_TEST_DIR test/cpp)

# print out compiler information
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# enable GPU if set
if (QUAKE_ENABLE_GPU)
    set(FAISS_ENABLE_GPU ON)
else ()
    set(FAISS_ENABLE_GPU OFF)
endif ()

if (APPLE)
    include_directories("/opt/homebrew/opt/openblas/include")
    set(FAISS_ENABLE_GPU OFF)
endif (APPLE)

add_compile_options(-march=native)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "Torch include dir: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "OpenMP Dir: ${OpenMP_CXX_INCLUDE_DIRS}")
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})

add_definitions(-DSIMSIMD_NATIVE_F16=0)
add_definitions(-DSIMSIMD_NATIVE_BF16=0)
add_subdirectory(${project_THIRD_PARTY_DIR})

file(GLOB_RECURSE project_SOURCES ${project_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME}
        SHARED
        ${project_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${project_INCLUDE_DIR}
        ${project_THIRD_PARTY_DIR}/SimSIMD/include/
        faiss)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# set linker libraries for linux and mac
if (APPLE)
    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            ${TORCH_LIBRARIES}
            faiss
            OpenMP::OpenMP_CXX
            -ffast-math
            -lpthread
            -fPIC
    )
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            ${TORCH_LIBRARIES}
            faiss_avx512
            OpenMP::OpenMP_CXX
            -ffast-math
            -lpthread
            -liomp5
            -lnuma
            -fPIC
    )
endif ()

target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})

# Python bindings
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
file(GLOB_RECURSE BINDINGS src/cpp/bindings/*.cpp)

pybind11_add_module(_bindings ${BINDINGS})
set_target_properties(_bindings PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(_bindings PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
add_custom_target(bindings)
add_dependencies(bindings _bindings)

# Determine all benchmark scripts
file(GLOB BENCHMARK_SOURCES "src/cpp/test/benchmark/*.cpp")

# Create an executable for each file
foreach(SOURCE ${BENCHMARK_SOURCES})
    # Get the filename without extension
    get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)

    # Add executable
    add_executable(${EXECUTABLE_NAME} ${SOURCE})

    # set linker libraries for linux and mac
    if (APPLE)
        target_link_libraries(${EXECUTABLE_NAME}
                PUBLIC
                ${TORCH_LIBRARIES}
                faiss
                OpenMP::OpenMP_CXX
                -ffast-math
                -lpthread
                -fPIC
        )
        target_include_directories(${EXECUTABLE_NAME}
                PUBLIC
                ${TORCH_INCLUDE_DIRS}
                ${project_INCLUDE_DIR}
                ${project_THIRD_PARTY_DIR}/SimSIMD/include/
                faiss)
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME})
    elseif (LINUX)
        target_link_libraries(${EXECUTABLE_NAME}
                PUBLIC
                ${TORCH_LIBRARIES}
                faiss_avx512
                OpenMP::OpenMP_CXX
                -ffast-math
                -lpthread
                -liomp5
                -lnuma
                -fPIC
        )
    endif ()

    target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endforeach()

# Tests
if (BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(${project_TEST_DIR})
endif ()