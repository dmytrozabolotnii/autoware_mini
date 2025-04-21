# Platform - leaderboard

## AutowareMiniRosAgent

This node provides a ROS autonomous agent interface to control the ego vehicle via a ROS stack in the CARLA simulator. It creates all necessary publishers and subscribers to interface with the CARLA simulator and the Autoware Mini stack. It initializes the stack, processes sensor data, and forwards control commands to the ego vehicle.

#### Parameters

| Name | Type | Default Value | Description |
| ----- | ----- | ------------- | ------------ |
| `/localization/use_custom_origin` | bool | `True` | Whether to use custom origin for UTM transformation. |
| `/localization/utm_origin_lat` | float | `0.0` | Latitude of UTM origin if using custom origin. |
| `/localization/utm_origin_lon` | float | `0.0` | Longitude of UTM origin if using custom origin. |
| `/carla_localization/use_transformer` | bool | `true` | Whether to use coordinate transformation between simulation and UTM coordinates. |
| `init_goal_delay` | float | `5` | Delay in seconds before publishing the first goal point. |
| `downsampling_interval` | int | `42` | Interval for downsampling the global path points. |

#### Subscribed Topics

| Name | Type | Description |
| ----- | ----- | ------------ |
| `/carla/ego_vehicle/vehicle_control_cmd` | `CarlaEgoVehicleControl` | Vehicle control commands from the autonomous stack. |

#### Published Topics

| Name | Type | Description |
| ----- | ----- | ------------ |
| `clock` | `rosgraph_msgs/Clock` | Simulation time. |
| `/carla/ego_vehicle/waypoints` | `nav_msgs/Path` | Global path waypoints for visualization. |
| `/move_base_simple/goal` | `geometry_msgs/PoseStamped` | Goal points for navigation. |
| `/carla/ego_vehicle/odometry` | `nav_msgs/Odometry` | Ego vehicle odometry. |
| `/carla/ego_vehicle/vehicle_status` | `carla_msgs/CarlaEgoVehicleStatus` | Current vehicle status including velocity, acceleration, and control commands. |
| `/carla/ego_vehicle/vehicle_info` | `carla_msgs/CarlaEgoVehicleInfo` | Vehicle physical properties including wheels, mass, and center of mass. |
| `/carla/map_file` | `std_msgs/String` | OpenDRIVE map data as string. |
| `/carla/world_info` | `carla_msgs/CarlaWorldInfo` | CARLA world information. |

Additionally, the node publishes various sensor data topics based on the configured sensors:
- Camera images: `<sensor_id>/image_raw` (Image) and `<sensor_id>/camera_info` (CameraInfo)
- LiDAR point clouds: `<sensor_id>/pointcloud` (PointCloud2)
- GNSS data: `<sensor_id>` (NavSatFix)
- IMU data: `<sensor_id>` (Imu)
