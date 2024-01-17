
cmake_minimum_required(VERSION 3.0.0)

project(tuto-ceres)

# Author: keir@google.com (Keir Mierle)
add_executable(tuto-ceres src/helloworld.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "helloworld")
#add_executable(tuto-ceres src/helloworld_numeric_diff.cpp)
#set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "helloworld_nim")
#add_executable(tuto-ceres src/helloworld_analytic_diff.cpp)
#set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "hellworld_ana")
#add_executable(tuto-ceres src/curve_fitting.cpp)
#set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "curve_fitting")

find_package(Ceres REQUIRED)

target_link_libraries(tuto-ceres PUBLIC
    ${CERES_LIBRARIES}
    )

target_include_directories(tuto-ceres PUBLIC
    ${CERES_INCLUDE_DIRS}
    )

# As this is a C file #including <math.h> we have to explicitly add the math
# library (libm). Although some compilers (dependent upon options) will accept
# the indirect link to libm via Ceres, at least GCC 4.8 on pure Debian won't.
add_subdirectory(src/pose_graph_2d)