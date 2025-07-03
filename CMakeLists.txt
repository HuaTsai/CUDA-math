cmake_minimum_required(VERSION 3.18)
project(CUDA-math LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

find_package(GTest REQUIRED)
include(GoogleTest)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")

add_executable(test_intersection 
    test/test_intersection.cu
)

set_property(TARGET test_intersection PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_intersection 
    ${CUDA_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_intersection PRIVATE src)

enable_testing()
gtest_discover_tests(test_intersection)
