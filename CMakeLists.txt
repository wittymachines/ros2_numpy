cmake_minimum_required(VERSION 3.5)
project(ros_numpy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_Package(tf2 REQUIRED)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_nose REQUIRED)
  #test_geometry.py  test_images.py  test_occupancygrids.py
  ament_add_nose_test(pointclouds test/test_pointclouds.py)
endif()

##############
ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_package()
