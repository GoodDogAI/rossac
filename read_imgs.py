import rosbag

import json
import argparse
import png
import os.path
import onnxruntime as rt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--camera_topic", default='/camera/infra2/image_rect_raw')
arg_parser.add_argument('--onnx', type=str, default='./yolov5s.onnx', help='onnx weights path')
arg_parser.add_argument("bags", nargs='+')
args = arg_parser.parse_args()

# class Vec3:
#     x = float
#     y = float
#     z = float

# class CmdVel:
#     angular = Vec3
#     linear = Vec3

sess = rt.InferenceSession(args.onnx)


for bag_file in args.bags:
    bag = rosbag.Bag(bag_file, 'r')
    seen = set()

    cmd_file = open(bag_file + ".cmd_vels", 'w')
    dynamixel_file = open(bag_file + ".dynamixel", 'w')

    for topic, msg, ts in bag.read_messages([args.camera_topic, '/cmd_vel', '/dynamixel_workbench/dynamixel_state']):
        #if topic not in seen:
            full_ts = ts.nsecs + ts.secs * 1000000000
            print("{} @ {}".format(topic, full_ts))
            if topic == args.camera_topic:
                print("{}x{} {}; row bytes: {}".format(msg.width, msg.height, msg.encoding, msg.step))
                print(type(msg.data))
                img_name = os.path.join(os.path.dirname(bag_file), 'imgs', '{}.png'.format(full_ts))
                if not os.path.isfile(img_name):
                    img = []
                    for i in range(0, len(msg.data), msg.step):
                        img.append(msg.data[i:i+msg.step])

                    img_mode = 'L' if "infra" in args.camera_topic else 'RGB'
                    png.from_array(img, mode=img_mode).save(img_name)

                # Save off the intermediate state which is passed as input to the MLP-SAC

                # Save off the reward score, for that image

            elif topic == '/dynamixel_workbench/dynamixel_state':
                json.dump({
                    "ts": full_ts,
                    "pan_state": msg.dynamixel_state[0].present_position,
                    "tilt_state": msg.dynamixel_state[1].present_position,
                }, dynamixel_file)
                dynamixel_file.write("\n")
            else:
                json.dump({
                    "ts": full_ts,
                    "linear": [msg.linear.x, msg.linear.y, msg.linear.z],
                    "angular": [msg.angular.x, msg.angular.y, msg.angular.z]
                }, cmd_file)
                cmd_file.write("\n")

            print()
            seen.add(topic)

    cmd_file.close()