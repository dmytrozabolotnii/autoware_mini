# Autoware Mini

### This is a fork of main autoware_mini repository for publishing the code for the article "FOV-RVO: Velocity Obstacle-based pedestrian motion predictor"

Autoware Mini is a minimalistic Python-based autonomy software inspired by [Autoware](https://www.autoware.org/). It is built on Python and ROS 1 to make it easy to get started and tinkering. Autoware Mini currently works on ROS Noetic (Ubuntu 20.04). The software is open-source with a friendly MIT license.

## TODO:

- [x] Upload implementation code with instructions
- [x] Share evaluation .bag dataset
- [ ] Include result processing jupyter notebooks

It is not production-level software, but aimed for teaching and research. At the same time we have validated the software with a real car in real traffic in the city of Tartu, Estonia.

## Architecture

![Autoware Mini diagram](images/diagram.png)

The key modules of Autoware Mini are:
* **[Localization](nodes/localization)** - determines vehicle position and speed. Can be implemented using GNSS, lidar positioning, visual positioning, etc.
* **[Obstacle detection](nodes/detection)** - produces detected objects based on lidar, radar or camera readings. Includes tracking and prediction.
* **[Traffic light detection](nodes/detection)** - produces status for stoplines, if they are green or red. Red stopline is like an obstacle for the local planner.
* **[Global planner](nodes/planning/global)** - given current position and destination determines the global path to the destination. Makes use of Lanelet2 map.
* **[Local planner](nodes/planning/local)** - given the global path and obstacles, plans a local path that avoids obstacles and respects traffic lights.
* **[Controller](nodes/control)** - follows the local path given by the local planner, matching target speeds at different points of trajectory.

Here are couple of (slightly outdated) short videos introducing the Autoware Mini features.

[![Autoware Mini planning simulator](https://img.youtube.com/vi/k3dOySPAYaY/mqdefault.jpg)](https://www.youtube.com/watch?v=k3dOySPAYaY&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=1 "Autoware Mini planning simulator")
[![Autoware Mini perception testing with SFA detector](https://img.youtube.com/vi/bn3G2WqHEYA/mqdefault.jpg)](https://www.youtube.com/watch?v=bn3G2WqHEYA&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=2 "Autoware Mini perception testing with SFA detector")
[![Autoware Mini perception testing with cluster detector](https://img.youtube.com/vi/OqKMQ5hUgn0/mqdefault.jpg)](https://www.youtube.com/watch?v=OqKMQ5hUgn0&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=3 "Autoware Mini perception testing with cluster detector")
[![Autoware Mini Carla testing with ground truth detector](https://img.youtube.com/vi/p8A05yQ1pfw/mqdefault.jpg)](https://www.youtube.com/watch?v=p8A05yQ1pfw&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=4 "Autoware Mini Carla testing with ground truth detector")
[![Autoware Mini Carla testing with cluster detector](https://img.youtube.com/vi/QEoPoBogIBc/mqdefault.jpg)](https://www.youtube.com/watch?v=QEoPoBogIBc&list=PLuQzXioASss3dJvI9kLvriGXMfQEYKXZO&index=5&t=2s "Autoware Mini Carla testing with cluster detector")

## Prerequisites

1. You should have ROS Noetic installed, follow the official instructions for [Ubuntu 20.04](http://wiki.ros.org/noetic/Installation/Ubuntu).

2. Some of the nodes need NVIDIA GPU, CUDA and cuDNN. At this point we suggest installing CUDA 11.8 for the best compatibility. **Notice that the default setup also runs without GPU.**

   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda=11.8.0-1 libcudnn8=8.9.7.29-1+cuda11.8
   sudo apt-mark hold cuda cuda-drivers
   ```
   If the above instructions installed/upgraded Nvidia drivers, please reboot your system before proceeding. If you have newer CUDA installed, but are happy to have it downgraded to 11.8, add `--allow-downgrades` to install command. Or you can choose to install `cuda-11.8` instead, which keeps the existing newer CUDA.

   In case the above instructions are out of date, follow the official [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html) installation instructions.

## Installation

1. Create workspace
   ```
   mkdir -p autoware_mini_ws/src
   cd autoware_mini_ws/src
   ```

2. Clone the repo
   ```
   git clone https://github.com/UT-ADL/autoware_mini.git
   ```

3. Install system dependencies (ignore the errors for missing Carla packages if not using Carla)

   ```
   rosdep update --include-eol-distros
   rosdep install --include-eol-distros --from-paths . --ignore-src -r -y
   ```

4. Install Python dependencies
   ```
   pip install -r autoware_mini/requirements.txt
   # only when planning to use GPU based clustering, long download
   pip install -r autoware_mini/requirements_cuml.txt
   pip install -r autoware_mini/requirements_torch.txt
   ```
You can use newer Pytorch version, compatibility with up to 2.7.1 was checked.

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

## Launching FOV-RVO against recorded bag from .bag dataset

Bags are provided at [S3 Object Store](https://docs.google.com/spreadsheets/d/1NyMl4oH4sRBBy59zweDUKbQBKxvu4mEOBFeWZL5IRvU/edit?usp=sharing). Place downloaded bag(s) at `data/bags`. Due to GDPR restrictions, the bags do not contain raw image data. Instead pre-processed head pose detections and gaze directions data are included in the bag files, and will be used by the FOV-RVO model automatically during bag playback when necessary.

To run the autonomy stack with FOV-RVO against the recorded bag run the following command and change "BAG_FILE_NAME" to one of the .bag files downloaded:

```
roslaunch autoware_mini start_bag.launch bag_file:="BAG_FILE_NAME" predictor:=pedestrianrvofovmap detector:=lidar_sfa map_name:=tartu_large loop:=false
```

There are different predictor parameters to run specific FOVRVO configurations described in the paper:

* `predictor:=pedestrianrvo`: Pure RVO implementation (no gaze direction data used)
* `predictor:=pedestrianrvofov`: RVO implementation with gaze direction constraint and variable responsibility (using pre-recorded gaze directions from the bag)
* `predictor:=pedestrianrvomap`: RVO implementation with map information integration (no gaze direction data used)
* `predictor:=pedestrianrvofovmap`: Combination of the above and main FOVRVO model (using pre-recorded gaze directions from the bag)
* `predictor:=pedestrianrvogaaddon`: Combined model of FOVRVO and [GATraj](https://github.com/mengmengliu1998/GATraj) model (using pre-recorded gaze directions from the bag)

To run Deep Learning pedestrian motion predictors instead against which the model is evaluated use the following commands. For these models, some parameters you can change including the amount of candidate predictions they output $k$ can be found in `config/detection.yaml` file under the `prediction` category. ($k$ is denoted in the file as `predictions_amount`)
* `predictor:=pedestrian`: [PECNet](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/PECNet) model
* `predictor:=pedestriansg`: [SGNet](https://github.com/ChuhuaW/SGNet.pytorch) model
* `predictor:=pedestrianga`: [GATraj](https://github.com/mengmengliu1998/GATraj) model

The detection topics in the bag file are remapped to dummy topic names and new detections are generated by the autonomy stack. The visualization will show online calculations of minDynADE/minDynFDE/MR/nonDAC metrics, and afterwards the results are saved under `data/results/prediction`, where the last line is the final result for the entire bag file evaluation.