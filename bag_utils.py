import rosbag
import png
import os
import numpy as np

from collections import defaultdict
from typing import DefaultDict, Any

import yolo_reward
from yolo_reward import get_onnx_sess, get_prediction, get_intermediate_layer

CAMERA_TOPIC = "/camera/infra2/image_rect_raw"


def _flatten(arr):
    return np.reshape(arr, -1)


def read_bag(bag_file: str,
             backbone_onnx_path: str,
             reward_func_name: str,
             reward_delay_ms: int, punish_backtrack_ms: int,
             wait_for_each_msg: bool = False) -> DefaultDict[str, Any]:
    bag = rosbag.Bag(bag_file, 'r')
    reward_func = getattr(yolo_reward, reward_func_name)

    entries = defaultdict(dict)
    received_topic = defaultdict(bool)

    ros_topics = [CAMERA_TOPIC,
                  '/dynamixel_workbench/dynamixel_state',
                  '/camera/accel/sample',
                  '/camera/gyro/sample',
                  '/head_feedback',
                  '/cmd_vel',
                  '/odrive_feedback',
                  '/reward_button',
                  '/vbus']

    for topic, msg, ts in bag.read_messages(ros_topics):
        full_ts = ts.nsecs + ts.secs * 1000000000

        received_topic[topic] = True
        if wait_for_each_msg and not all(received_topic[topic] for topic in ros_topics):
            continue

        if topic == CAMERA_TOPIC:
            img = []
            for i in range(0, len(msg.data), msg.step):
                img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))

            assert "infra" in CAMERA_TOPIC, "Expecting mono infrared images only right now"

            # Convert list of byte arrays to numpy array
            image_np = np.array(img)
            pred = get_prediction(get_onnx_sess(backbone_onnx_path), image_np)
            intermediate = get_intermediate_layer(pred)
            reward = reward_func(pred)

            if np.isnan(intermediate).any():
                print(f"Invalid YOLO output in bag: {bag_file} at ts {ts}")
                img_mode = 'L' if "infra" in CAMERA_TOPIC else 'RGB'
                png.from_array(img, mode=img_mode).save(os.path.join(os.path.dirname(bag_file), f"error_{os.path.basename(bag_file)}_{ts}.png"))
                continue

            entries["yolo_intermediate"][full_ts] = _flatten(intermediate)
            entries["reward"][full_ts + reward_delay_ms * 1000000] = reward
        elif topic == '/reward_button':
            entries["punishment"][full_ts + punish_backtrack_ms * 1000000] = np.array([msg.data])
        elif topic == '/dynamixel_workbench/dynamixel_state':
            entries["dynamixel_cur_state"][full_ts] = np.array([msg.dynamixel_state[0].present_position,
                                                                msg.dynamixel_state[1].present_position])
        elif topic == "/head_feedback":
            entries["dynamixel_command_state"][full_ts] = np.array([msg.pan_command,
                                                                    msg.tilt_command])
        elif topic == "/cmd_vel":
            entries["cmd_vel"][full_ts] = np.array([msg.linear.x,
                                                    msg.angular.z])
        elif topic == "/camera/accel/sample":
            entries["head_accel"][full_ts] = np.array([msg.linear_acceleration.x,
                                                       msg.linear_acceleration.y,
                                                       msg.linear_acceleration.z])
        elif topic == "/camera/gyro/sample":
            entries["head_gyro"][full_ts] = np.array([msg.angular_velocity.x,
                                                      msg.angular_velocity.y,
                                                      msg.angular_velocity.z])
        elif topic == "/odrive_feedback":
            entries["odrive_feedback"][full_ts] = np.array([msg.motor_vel_actual_0,
                                                            msg.motor_vel_actual_1,
                                                            msg.motor_vel_cmd_0,
                                                            msg.motor_vel_cmd_1])
        elif topic == "/vbus":
            entries["vbus"][full_ts] = np.array([msg.data])
        else:
            raise KeyError("Unexpected rosbag topic")

    return entries