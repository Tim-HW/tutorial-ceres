
cmake_minimum_required(VERSION 3.0.0)

project(tuto-ceres)

# List of executable names
set(EXECUTABLE_NAMES
    helloworld
    powell
    curve_fitting
    helloworld_analytic_diff
    helloworld_numeric_diff
    # Add more executable names if needed
)

# Loop over the executable names and add_executable for each
foreach(EXECUTABLE ${EXECUTABLE_NAMES})
    add_executable(${EXECUTABLE}
        src/${EXECUTABLE}.cpp
    )
    target_include_directories(${EXECUTABLE} PUBLIC
    ${CERES_INCLUDE_DIRS}
    )
    target_link_libraries(${EXECUTABLE} PUBLIC
    ${CERES_LIBRARIES}
    )
endforeach()


# Author: keir@google.com (Keir Mierle)
#add_executable(helloworld src/helloworld.cpp)


find_package(Ceres REQUIRED)


# Loop over the executable names and add_executable for each
foreach(EXECUTABLE ${EXECUTABLE_NAMES})
   
    target_include_directories(${EXECUTABLE} PUBLIC
    ${CERES_INCLUDE_DIRS}
    )
    target_link_libraries(${EXECUTABLE} PUBLIC
    ${CERES_LIBRARIES}
    )
endforeach()


# As this is a C file #including <math.h> we have to explicitly add the math
# library (libm). Although some compilers (dependent upon options) will accept
# the indirect link to libm via Ceres, at least GCC 4.8 on pure Debian won't.
add_subdirectory(src/pose_graph_2d)
