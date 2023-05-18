# Autoware Mini

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

## Installation

You should have ROS Noetic installed, follow the official instructions for [Ubuntu 20.04](http://wiki.ros.org/noetic/Installation/Ubuntu) or [RoboStack](https://robostack.github.io/GettingStarted.html).

1. Create workspace
   ```
   mkdir -p autoware_mini_ws/src
   cd autoware_mini_ws/src
   ```

2. Clone the repos
   ```
   git clone git@gitlab.cs.ut.ee:autonomous-driving-lab/autoware_mini.git
   # not needed for planner simulation
   git clone git@gitlab.cs.ut.ee:autonomous-driving-lab/autoware.ai/local/vehicle_platform.git
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
   source $HOME/autoware_mini_ws/devel/setup.bash
   ```

## Launching planner simulation

Planner simulation is very lightweight and has the least dependencies.

```
roslaunch autoware_mini start_sim.launch
```

You should see Rviz window with a default map. To start driving you need to give the vehicle initial position with 2D Pose Estimate button and goal using 2D Nav Goal button. Obstacles can be placed or removed with Publish Point button.

## Launching against recorded bag

Running the autonomy stack against recorded bag file is a convenient way to test the perception module. The example bag file can be downloaded from [here](https://owncloud.ut.ee/owncloud/s/fdyJa5789oJKAgE) and it shoud be saved to the `data/bags` directory.

```
roslaunch autoware_mini start_bag.launch bag_file:=<name of the bag file in data/bags directory>
```

The detection topics in bag are remapped to dummy topic names and new detections are generated by the autonomy stack.

## Launching Carla simulation

### Download Carla + Tartu map (Skip if already done)

1. Download [Carla 0.9.13](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz) and extract it with `tar xzvf CARLA_0.9.13.tar.gz`. We will call this extracted folder `<CARLA ROOT>`.
2. Download [Tartu.tar.gz](https://drive.google.com/file/d/10CHEOjHyiLJgD13g6WwDZ2_AWoLasG2F/view?usp=share_link).
3. Copy `Tartu.tar.gz` inside the `Import` folder under `<CARLA ROOT>` directory.
4. Run `./ImportAssets.sh` from the `<CARLA ROOT>` directory. This will install the Tartu map. (You can now delete the `Tartu.tar.gz` file from the `Import` folder.)
5. Since we will be referring to `<CARLA ROOT>` a lot, let's export it as an environment variable. Make sure to replace the path where Carla is extracted.

   ```
   export CARLA_ROOT=$HOME/path/to/carla
   ```

6. Now, enter the following command. (**NOTE:** Here we assume that `CARLA_ROOT`  was set from the previous command.)
   ```
   export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla
   ```
   **Note:** It will be convenient if the above variables are automatically exported whenever you open a terminal. Putting above exports in `~/.bashrc` will reduce the hassle of exporting everytime.

### Launch instructions

1. In a new terminal, (assuming enviornment variables are exported) run Carla simulator by entering the following command.

   ```
   $CARLA_ROOT/CarlaUE4.sh -prefernvidia -quality-level=Low
   ```
#### Launch using ground-truth detection:
2. In a new terminal, (assuming enviornment variables are exported) run the following command. This runs Tartu environment of Carla with minimal sensors and our autonomy stack. The detected objects come from Carla directly.

   ```
   roslaunch autoware_mini start_carla.launch
   ```
#### OR
#### Launch using lidar based detector:
2. In a new terminal, (assuming enviornment variables are exported) run the following command. This runs Tartu environment of Carla with lidar sensors and our autonomy stack. The detection is performed using lidar-based cluster detector.

   ```
   roslaunch autoware_mini start_carla.launch detector:=cluster
   ```
## Launching in Lexus

```
roslaunch autoware_mini start_lexus.launch
```
