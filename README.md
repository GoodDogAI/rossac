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
 - [ ] Write a script that can take an SAC onnx file, and then run it end-to-end and output the actions. Are those action outputs reasonable on the reply buffer, do they match on both the offline and online bot code running TensorRT?
 - [ ] In the Spinningup ai examples, they calculate a Reward-to-go for an episode, discounting the reward for each step in that episode. But our current code just uses the instantaneous reward at each step.
 - [ ] Make observation vector richer (actual wheel speed, head acceleration, etc)
 - [ ] Make action space a "rate of change" instead of absolute
 - [ ] Add WANDB support for tracking training runs.
