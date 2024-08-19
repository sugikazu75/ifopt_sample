cmake_minimum_required(VERSION 3.0.2)
project(ifopt_sample)

find_package(catkin REQUIRED COMPONENTS
  ifopt
  )
find_package(Eigen3 REQUIRED)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(sample1 src/sample1.cpp)
target_link_libraries(sample1 ${catkin_LIBRARIES})

