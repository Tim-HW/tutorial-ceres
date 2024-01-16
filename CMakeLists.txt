cmake_minimum_required(VERSION 3.0.0)
project(test VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(helloworld main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# external libs
find_package(Ceres REQUIRED)

target_include_directories(helloworld
  PRIVATE
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(helloworld
  PRIVATE
    ${CERES_LIBRARIES}
)