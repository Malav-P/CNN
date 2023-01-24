get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)

include(CMakeFindDependencyMacro)

find_dependency(Boost REQUIRED)
find_dependency(nlohmann_json REQUIRED)

include(${SELF_DIR}/cmake/cnn.cmake)
