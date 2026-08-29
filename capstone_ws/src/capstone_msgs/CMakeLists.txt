cmake_minimum_required(VERSION 3.8)
project(capstone_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# All 17 frozen interface types (count corrected from a stale "15" in the spec).
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ReceiverState.msg"
  "msg/ReceiverStateArray.msg"
  "msg/EmitterTruth.msg"
  "msg/TruthScene.msg"
  "msg/RfObservation.msg"
  "msg/RfObservationSet.msg"
  "msg/BurstAssociationTruth.msg"
  "msg/BurstAssociation.msg"
  "msg/BurstAssociationSet.msg"
  "msg/Embedding.msg"
  "msg/FingerprintAssignment.msg"
  "msg/FingerprintSet.msg"
  "msg/TrackState.msg"
  "msg/TrackArray.msg"
  "msg/PathPrediction.msg"
  "msg/FormationEstimate.msg"
  "msg/Narration.msg"
  "msg/NodeHealth.msg"
  "msg/SceneUpdate.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
