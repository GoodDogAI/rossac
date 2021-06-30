# rossac

Features:
 - Take robot camera image from a rosbag and calculate a reward based on an intermediate YOLOV5 layer
 - Interpolate between the dynamixel and velocity commands between frames to get consistent snapshots of the robot state at a given timestamp
 - Train a SAC model given rosbags as preprocessed above

Status: Not working, in test-mode on a real robot, all of the outputs are either very close to zero, or just constants

TODOs:
 - [ ] Make sure that the intermediate layers for a given camera input are the same when we re-calculate them on the training PC, or else save the raw intermediate YOLO layers into the bags themselves
 - [ ] Verify the squashed MLP gaussian is outputting correctly (https://github.com/thu-ml/tianshou/pull/333)
 - [ ] What's the point of the `end_of_episode` flag. It changes how the Bellman backup is calculated, but our episodes are more or less continuous 
 - [ ] Is the ONNXNormal distribution that we wrote perfectly matching the Pytorch original version? Write unit tests.
