cmake_minimum_required(VERSION 3.16)
project(CNN)

set(CMAKE_CXX_STANDARD 14)

include_directories(/opt/homebrew/include/)

add_executable(linalg_test testing/linalg_test.cpp)
add_executable(convolution_test testing/convolution_test.cpp)
add_executable(maxpool_test testing/maxpool_test.cpp)
add_executable(meanpool_test testing/meanpool_test.cpp)
add_executable(linear_test testing/linear_test.cpp)
add_executable(model_test testing/model_test.cpp mnist/read_mnist_impl.cpp)
add_executable(readmnist_test testing/readmnist_test.cpp mnist/read_mnist_impl.cpp)
