import functools
import os.path

from collections import defaultdict

import numpy as np
import onnxruntime as rt
import pandas as pd
import png
import pyarrow as pa
import rosbag

import yolo_reward

from interpolate import interpolate_events
from yolo_reward import get_prediction, get_intermediate_layer

DATAFRAME_COLUMNS = [
    # The first entry is the one that we will interpolate over
    "yolo_intermediate",
    "reward",

    # pan, tilt in steps, (1024 steps = 300 degrees)
    "dynamixel_cur_state",
    "dynamixel_command_state",

    # commanded forward speed, rotational speed
    "cmd_vel",

    # Each wheels actual speed in rotations per second
    "odrive_feedback",

    # head gyro rate, in radians per second
    "head_gyro",

    # head acceleration, in meters per second
    "head_accel",

    # robot bus voltage, in Volts
    "vbus",

    # indicates reward from stop button (0 = unpressed, -1 = pressed)
    "punishment",
]

def read_bag(bag_file: str, backbone_onnx_path: str,
             cache_dir: str,
             camera_topic: str,
             reward_func_name: str,
             reward_delay_ms: int, punish_backtrack_ms: int) -> pd.DataFrame:
    print(f"Opening {bag_file}")
    bag_cache_name = os.path.join(cache_dir, f"{os.path.basename(bag_file)}_{reward_func_name}_+{reward_delay_ms}ms_-{punish_backtrack_ms}ms.arrow")

    try:
        return _read_mmapped_bag(bag_cache_name)
    except IOError:
        write_bag_cache(bag_file, bag_cache_name, backbone_onnx_path,
                        camera_topic=camera_topic,
                        reward_func_name=reward_func_name,
                        reward_delay_ms=reward_delay_ms, punish_backtrack_ms=punish_backtrack_ms)
        return _read_mmapped_bag(bag_cache_name)


def _read_mmapped_bag(bag_cache_name: str) -> pd.DataFrame:
    source = pa.memory_map(bag_cache_name, "r")
    table = pa.ipc.RecordBatchFileReader(source).read_all()
    return table.to_pandas()


def write_bag_cache(bag_file: str, bag_cache_path: str, backbone_onnx_path: str,
                    camera_topic: str,
                    reward_func_name: str,
                    reward_delay_ms: int, punish_backtrack_ms: int):
    bag = rosbag.Bag(bag_file, 'r')
    entries = defaultdict(dict)
    reward_func = getattr(yolo_reward, reward_func_name)

    # If you are the first bag in a series, don't output any entries until you received one message from each channel
    # This is because it can take some time for everything to fully initialize (ex. if building tensorrt models),
    # and we don't want bogus data points.
    wait_for_each_msg = bag_file.endswith("_0.bag")
    received_topic = defaultdict(bool)
    ros_topics = [camera_topic,
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

        if topic == camera_topic:
            img = []
            for i in range(0, len(msg.data), msg.step):
                img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))

            assert "infra" in camera_topic, "Expecting mono infrared images only right now"

            # Convert list of byte arrays to numpy array
            image_np = np.array(img)
            pred = get_prediction(get_onnx_sess(backbone_onnx_path), image_np)
            intermediate = get_intermediate_layer(pred)
            reward = reward_func(pred)

            if np.isnan(intermediate).any():
                print(f"Invalid YOLO output in bag: {bag_file} at ts {ts}")
                img_mode = 'L' if "infra" in camera_topic else 'RGB'
                png.from_array(img, mode=img_mode).save(f"{os.path.basename(bag_file)}_{ts}.png")
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

    interpolated = interpolate_events(entries["yolo_intermediate"], [entries[key] for key in DATAFRAME_COLUMNS[1:]],
                                      max_gap_ns=1000 * 1000 * 1000)

    df = pd.DataFrame.from_records([[ts, event, *interps] for (ts, event, interps) in interpolated],
                                    columns=["ts", *DATAFRAME_COLUMNS], index="ts")

    # Convert from pandas to Arrow
    table = pa.Table.from_pandas(df)

    # Write out to file
    with pa.OSFile(bag_cache_path, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

@functools.lru_cache()
def get_onnx_sess(onnx_path: str) -> rt.InferenceSession:
    print("Starting ONNX inference session")
    return rt.InferenceSession(onnx_path)

def _flatten(arr):
    return np.reshape(arr, -1)