cmake_minimum_required(VERSION 3.15)
project(CNN)


add_subdirectory(include)

#option(BUILD_EXAMPLES "" ON)

#if (BUILD_EXAMPLES)
#	message(STATUS "Building examples ....")
#	add_subdirectory(examples/CNN-example)
#endif()


install(FILES cmake/CNN-config.cmake
	DESTINATION cmake)

install(EXPORT CNN
	DESTINATION cmake)
