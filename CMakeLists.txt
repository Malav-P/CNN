cmake_minimum_required(VERSION 3.15)
project(CNN)


add_subdirectory(include)

install(FILES cmake/CNN-config.cmake
	DESTINATION cmake)

install(EXPORT CNN
	DESTINATION cmake)
