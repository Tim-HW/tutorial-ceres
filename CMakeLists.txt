cmake_minimum_required(VERSION 3.0.0)
project(ceres-tuto VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(ceres-tuto src/2D_pose_graph/2d_slam.cpp)

add_subdirectory(src/2D_pose_graph)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# external libs
find_package(Ceres REQUIRED)

target_include_directories(ceres-tuto
  PRIVATE
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(ceres-tuto
  PRIVATE
    ${CERES_LIBRARIES}
)