# FoundationPoseClient

## Prerequists
- ROS melodic
- Python 3.9

## Install
```
# Create a CONDA env
conda create -n foundationpose-client python==3.9

# Activate CONDA
conda activate foundationpose-client

# Install dependencies
python -m pip install -r requirements.txt
```

## Run
Run the Server first.
### Simple demo example
```
python example.py
```
### Encapsulated Class
```
python realsense_sam_foundationpose_tracker.py
```
### ROS Publisher
Run Realsense Camera in ROS:
```
roslaunch realsense2_camera rs_rgbd.launch
```
Run the Publisher:
```
rosrun tutorial pose_tracker_ros_publisher.py
```

## Use
### Mask Selector
This is a GUI wrapper of SAM2 with click prompts. The object names of the mask to be prompted is displayed as the title. Use mouse and keyboard to interact:
- Mouse
    - `Left click`: select a positive point prompt and display a green point.
    - `Right click`: select a negative point prompt and display a red point.
    - After each prompt, the blue mask will be updated and (re)shown.
- Keyboard
    - `Space`: confirm the current mask for the object.
    - `R`:[R]efresh and clean all the masks and prompts.
    - `N`: skip current object and move to [N]ext object.
    - `ESC`: close the GUI.
    - The masks will be displayed after all object masks are selected.

### Bounding Box Selector
This is a GUI for selecting a bounding box (i.e. rectangular mask) manually. 
- Mouse
    - `Left press, drag, and release`: (re)select a rectangular area.
- Keyboard
    - `Space`: confirm the current mask for the object.
    - `R`:[R]efresh and clean all the masks and prompts.
    - `N`: skip current object and move to [N]ext object.
    - `ESC`: close the GUI.
    - The masks will be displayed after all object masks are selected.

You can change the mask selectors in `pose_tracker_ros_publisher.py`: If the parameter `sam_checkpoint` is `None`, the bounding box selector is used.

### ROS
The Publisher will publish the poses of a object to topic `/pose_tracker/[object name]` and the data type is `geometry_msgs/PoseStamped`. 