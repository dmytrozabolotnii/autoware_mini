# Autoware Mini

### This is fork of main autoware_mini repository for publishing the code for the letter "Pedestrian motion prediction evaluation for urban autonomous driving"

Autoware Mini is a minimalistic Python-based autonomy software. It is built on Python and ROS 1 to make it easy to get started and tinkering. It uses Autoware messages to define the interfaces between the modules, aiming to be compatible with [Autoware](https://www.autoware.org/). Autoware Mini currently works on ROS Noetic (Ubuntu 20.04 and through [Conda RoboStack](https://robostack.github.io/) also on many other Linux versions). The software is open-source with a friendly MIT license.

## Goals

Our goals with the Autoware Mini were:
* easy to get started with --> minimal amount of dependencies
* simple and pedagogical --> simple Python nodes and ROS 1
* easy to implement machine learning based approaches --> Python

It is not production-level software, but aimed for teaching and research. At the same time we have validated the software with a real car in real traffic in the city center of Tartu, Estonia.

## Architecture

![Autoware Mini diagram](images/diagram.png)

The key modules of Autoware Mini are:
* **Localization** - determines vehicle position and speed. Can be implemented using GNSS, lidar positioning, visual positioning, etc.
* **Global planner** - given current position and destination determines the global path to the destination. Makes use of Lanelet2 map.
* **Obstacle detection** - produces detected objects based on lidar, radar or camera readings. Includes tracking and prediction.
* **Traffic light detection** - produces status for stoplines, if they are green or red. Red stopline is like an obstacle for the local planner.
* **Local planner** - given the global path and obstacles, plans a local path that avoids obstacles and respects traffic lights.
* **Follower** - follows the local path given by the local planner, matching target speeds at different points of trajectory.

Here are couple of short videos introducing the Autoware Mini features.

[![Autoware Mini planning simulator](https://img.youtube.com/vi/k3dOySPAYaY/mqdefault.jpg)](https://www.youtube.com/watch?v=k3dOySPAYaY&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=1 "Autoware Mini planning simulator")
[![Autoware Mini perception testing with SFA detector](https://img.youtube.com/vi/bn3G2WqHEYA/mqdefault.jpg)](https://www.youtube.com/watch?v=bn3G2WqHEYA&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=2 "Autoware Mini perception testing with SFA detector")
[![Autoware Mini perception testing with cluster detector](https://img.youtube.com/vi/OqKMQ5hUgn0/mqdefault.jpg)](https://www.youtube.com/watch?v=OqKMQ5hUgn0&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=3 "Autoware Mini perception testing with cluster detector")
[![Autoware Mini Carla testing with ground truth detector](https://img.youtube.com/vi/p8A05yQ1pfw/mqdefault.jpg)](https://www.youtube.com/watch?v=p8A05yQ1pfw&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=4 "Autoware Mini Carla testing with ground truth detector")
[![Autoware Mini Carla testing with cluster detector](https://img.youtube.com/vi/QEoPoBogIBc/mqdefault.jpg)](https://www.youtube.com/watch?v=QEoPoBogIBc&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=5&t=2s "Autoware Mini Carla testing with cluster detector")

## Prerequisites

1. You should have ROS Noetic installed, follow the official instructions for [Ubuntu 20.04](http://wiki.ros.org/noetic/Installation/Ubuntu) or [RoboStack](https://robostack.github.io/GettingStarted.html).

2. Some of the nodes need NVIDIA GPU, CUDA and cuDNN. At this point we suggest installing both the latest CUDA _and_ CUDA 11.8, which seems to be needed by the ONNX Runtime. **Notice that the default setup also runs without GPU.**

   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda cuda-11-8 libcudnn8
   ```

   In case the above instructions are out of date, follow the official [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) installation instructions.

## Installation

1. Create workspace
   ```
   mkdir -p autoware_mini_ws/src
   cd autoware_mini_ws/src
   ```

2. Clone the repos
   ```
   git clone https://github.com/UT-ADL/autoware_mini.git
   # not needed for the simplest planner simulation
   git clone https://github.com/UT-ADL/vehicle_platform.git
   # if using Carla simulation
   git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git carla_ros_bridge
   ```

3. Install system dependencies

   ```
   rosdep update
   rosdep install --from-paths . --ignore-src -r -y
   ```

4. Install Python dependencies
   ```
   pip install -r autoware_mini/requirements.txt
   pip install -r autoware_mini/requirements_cuml.txt
   pip install -r autoware_mini/requirements_torch.txt
   ```

5. Build the workspace
   ```
   cd ..
   catkin build
   ```

6. Source the workspace environment
   ```
   source devel/setup.bash
   ```
   As this needs to be run every time before launching the software, you might want to add something similar to the following line to your `~/.bashrc`.
   ```
   source ~/autoware_mini_ws/devel/setup.bash
   ```

## Launching against recorded bag from pedestrian prediction dataset

Bags are provided at [S3 Object Store](https://docs.google.com/spreadsheets/d/10CErfxRukFvESacOKBlnnJ5M4ltwxJ05pC5Z6f4HmW0/edit?usp=sharing). Place downloaded bag(s) at `data/bags`

To run the autonomy stack against recorded bag with a preset pedestrian prediction algorithm run the following command and change "BAG_FILE_NAME" to one of the .bag files downloaded:

```
roslaunch autoware_mini start_bag.launch bag_file:="BAG_FILE_NAME" predictor:="PREDICTOR_NAME" detector:=lidar_sfa map_name:=tartu_large loop:=false
```

There are 5 predictor algorithms available:

* `predictor:=naivecv`: Constant Velocity model
* `predictor:=pedestrian`: [PECNet](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/PECNet) model
* `predictor:=pedestriansg`: [SGNet](https://github.com/ChuhuaW/SGNet.pytorch) model
* `predictor:=pedestrianga`: [GATraj](https://github.com/mengmengliu1998/GATraj) model
* `predictor:=pedestrianmuse`: [MUSE-VAE](https://github.com/ml1323/musevae) model

The detection topics in bag are remapped to dummy topic names and new detections are generated by the autonomy stack. The visualization will show online calculations of minDynADE and minDynFDE metrics, and afterwards the results are saved under `data/results/prediction`. Some parameters you can change can be found in `config/detection.yaml` file under `prediction` category.
