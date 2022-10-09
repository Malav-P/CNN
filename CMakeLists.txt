cmake_minimum_required(VERSION 3.16)
project(CNN LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)


include_directories(/usr/local/cuda-11.7/include)


add_executable(cuda_test cuda_test.cu)


add_executable(linalg_test testing/linalg_test.cu)
add_executable(progress_bar_test testing/progress_bar_test.cu)
add_executable(convolution_test testing/convolution_test.cu UnifiedMemClass/managed.hxx)

set_target_properties(convolution_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
add_executable(maxpool_test testing/maxpool_test.cu)

set_target_properties(maxpool_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

add_executable(meanpool_test testing/meanpool_test.cu)
add_executable(linear_test testing/linear_test.cu)
add_executable(model_test testing/model_test.cu mnist/read_mnist_impl.cu)

set_target_properties(model_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
add_executable(readmnist_test testing/readmnist_test.cu mnist/read_mnist_impl.cu)