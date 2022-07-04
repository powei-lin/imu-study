cmake_minimum_required(VERSION 3.0.0)
project(imu-study VERSION 0.1.0)

add_subdirectory(thirdparty/basalt-headers)

add_executable(generate_imu_data generate_imu_data.cpp)
target_link_libraries(generate_imu_data basalt-headers)
target_include_directories(generate_imu_data PUBLIC thirdparty)
