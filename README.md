# HOI_taskPriority - ROS Package
(By: Raul Musito, Carlos Pazos, Samantha Caballero)


## Running the Simulation

To run the simulation using the provided commands, follow these steps:

1. Launch the Stonefish Simulation + Rviz :
   ```bash
   roslaunch hoi_taskpriority hoi.launch
   ```

2. In seperated terminal window, run the following node executes the Task Priority Control Algorithm with the mobile base and arm kinematics
   ```bash
   rosrun hoi_taskpriority TP_control_node.py
   ```
    This command will execute the "controller" node responsible for the ... process.

3. In another terminal, the following node will run the Aruco Detector node, which is in charge of receiving the XYZ position of detected aruco markers and publish it as a topic:
   ```bash
   rosrun hoi_taskpriority aruco_pose_detector_node.py
   ```
   
4. Finally, run the behavior tree node, which puts all the tasks together and ticks them as behaviors to run the whole operation:
   ```bash
   rosrun hoi_taskpriority behaviour_tree_node.py
   ```
