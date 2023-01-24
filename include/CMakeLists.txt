project(CNN)

add_library(cnn
		INTERFACE
		#	${CNN_INTERFACE_HEADERS}
		)

find_package(Boost REQUIRED)
find_package(nlohmann_json 3.7.3 REQUIRED)

if (APPLE)
	set(BLA_VENDOR Apple)
	target_compile_definitions(cnn INTERFACE APPLE)
else()
	set(BLA_VENDOR OpenBLAS)
	target_compile_definitions(cnn INTERFACE OTHER)
endif()

find_package(BLAS REQUIRED)

#FILE(GLOB_RECURSE CNN_INTERFACE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx")

include(GNUInstallDirs)


target_link_libraries(cnn
	INTERFACE
	Boost::boost
	nlohmann_json::nlohmann_json
		BLAS::BLAS
)

target_include_directories(cnn
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

#Install

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnn
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

install(TARGETS cnn EXPORT CNN
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cnn/cmake)



