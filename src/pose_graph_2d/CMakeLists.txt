cmake_minimum_required(VERSION 3.0.0)

project(pose_graph_2d)
# Author: vitus@google.com (Michael Vitus)

#if (GFLAGS)
  add_executable(pose_graph_2d
    angle_manifold.h
    normalize_angle.h
    pose_graph_2d.cc
    pose_graph_2d_error_term.h
    types.h)
  target_link_libraries(pose_graph_2d PRIVATE Ceres::ceres gflags)
#endif (GFLAGS)

find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
