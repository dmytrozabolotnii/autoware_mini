cmake_minimum_required(VERSION 2.8.12...3.25.2)
project(autoware_mini)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  lanelet2_core
  lanelet2_routing
  lanelet2_traffic_rules
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

find_package(Boost REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Development)

catkin_python_setup()

add_message_files(
  FILES
  ControlCommand.msg
  DetectedObject.msg
  DetectedObjectArray.msg
  Gear.msg
  LampCmd.msg
  Path.msg
  PathArray.msg
  TrafficLightResult.msg
  TrafficLightResultArray.msg
  VehicleCmd.msg
  VehicleStatus.msg
  Waypoint.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  LIBRARIES custom_routing_cost
  CATKIN_DEPENDS roscpp lanelet2_core lanelet2_routing lanelet2_traffic_rules std_msgs sensor_msgs geometry_msgs message_runtime 
)

include_directories(${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
add_library(custom_routing_cost MODULE src/custom_routing_cost/custom_routing_cost.cpp)
target_link_libraries(custom_routing_cost ${catkin_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(custom_routing_cost PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS custom_routing_cost
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

add_custom_command(TARGET custom_routing_cost POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
        $<TARGET_FILE:custom_routing_cost> 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/custom_routing_cost/custom_routing_cost.so
)
