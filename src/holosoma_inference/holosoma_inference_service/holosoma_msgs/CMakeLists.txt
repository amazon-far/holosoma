cmake_minimum_required(VERSION 3.8)
project(holosoma_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CmdExoskeleton.msg"
  "msg/Cmd3pt.msg"
  "msg/CmdSMPLH.msg"
  "msg/CmdDense.msg"
  "msg/Heartbeat.msg"
  "msg/Observation.msg"
  "msg/Action.msg"
  "msg/PolicyMetadata.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_package()
