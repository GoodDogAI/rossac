import rosbag

import json
import argparse
import png
import os.path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--camera_topic", default='/camera/infra2/image_rect_raw')
arg_parser.add_argument("bags", nargs='+')
args = arg_parser.parse_args()

# class Vec3:
#     x = float
#     y = float
#     z = float

# class CmdVel:
#     angular = Vec3
#     linear = Vec3
for bag_file in args.bags:
    bag = rosbag.Bag(bag_file, 'r')
    seen = set()

    cmd_file = open(bag_file + ".cmd_vels", 'w')

    for topic, msg, ts in bag.read_messages([args.camera_topic, '/cmd_vel']):
        #if topic not in seen:
            full_ts = ts.nsecs + ts.secs * 1000000000
            print("{} @ {}".format(topic, full_ts))
            if topic == args.camera_topic:
                print("{}x{} {}; row bytes: {}".format(msg.width, msg.height, msg.encoding, msg.step))
                print(type(msg.data))
                img_name = 'imgs/{}.png'.format(full_ts)
                if not os.path.isfile(img_name):
                    img = []
                    for i in range(0, len(msg.data), msg.step):
                        img.append(msg.data[i:i+msg.step])

                    img_mode = 'L' if "infra" in args.camera_topic else 'RGB'
                    png.from_array(img, mode=img_mode).save(img_name)
            else:
                print(dir(msg))
                cmd_file.write('{{ "ts": "{}", "linear": [{},{},{}], "angular": [{},{},{}] }}\n'
                                .format(full_ts, msg.linear.x, msg.linear.y, msg.linear.z,
                                                msg.angular.x, msg.angular.y, msg.angular.z))
            print()
            seen.add(topic)

    cmd_file.close()