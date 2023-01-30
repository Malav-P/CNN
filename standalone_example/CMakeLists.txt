

cmake_minimum_required(VERSION 3.16)
project("standalone_example")

set(CMAKE_CXX_STANDARD 14)

find_package(CNN REQUIRED)

set(MNIST_TEST  ${CMAKE_CURRENT_SOURCE_DIR}/mnist/mnist_test.csv)
set(MNIST_TRAIN ${CMAKE_CURRENT_SOURCE_DIR}/mnist/mnist_train.csv)

if(NOT EXISTS ${MNIST_TEST})
	message(STATUS "NOTICE-- mnist_test.csv file not found, downloading now to ${CMAKE_CURRENT_SOURCE_DIR}/mnist/")

	file(DOWNLOAD https://pjreddie.com/media/files/mnist_test.csv ${MNIST_TEST})
endif()

if(NOT EXISTS ${MNIST_TRAIN})
	message(STATUS "NOTICE-- mnist_train.csv file not found, downloading now to ${CMAKE_CURRENT_SOURCE_DIR}/mnist/")

	file(DOWNLOAD https://pjreddie.com/media/files/mnist_train.csv ${MNIST_TRAIN})

endif()

set(CMAKE_CXX_FLAGS "-O3 -ggdb")

add_executable(model_test models/model_test.cpp mnist/read_mnist_impl.cpp)
target_link_libraries(model_test PRIVATE cnn)

add_executable(model1 models/model1.cpp mnist/read_mnist_impl.cpp)
target_link_libraries(model1 PRIVATE cnn)

add_executable(model2 models/model2.cpp mnist/read_mnist_impl.cpp)
target_link_libraries(model2 PRIVATE cnn)

#add_executable(arraytest array_test.cpp)
#target_link_libraries(arraytest PRIVATE cnn)


