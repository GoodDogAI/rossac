import argparse
import os.path
import sys

# requires https://github.com/ArmyOfRobots/yolov5 to be cloned in ..\YOLOv5
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'YOLOv5'))

import torch

from bot_env import RobotEnvironment
from sac import SoftActorCritic
import models

def prepare_backbone(backbone):
    backbone.eval()

    # Don't fuse layers, it won't work with torchscript exports
    #model.fuse()

    for param in backbone.parameters():
        param.requires_grad = False
    
    # from https://stackoverflow.com/a/52548419/231238
    # cut last layer
    prepared = torch.nn.Sequential(*(list(backbone.model.children())[:-1]))
    assert(list(prepared.children())[-1].type == 'models.common.BottleneckCSP')

    return prepared

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    opt = parser.parse_args()

    backbone = torch.load(opt.model)['model'].float()
    backbone = prepare_backbone(backbone)

    # every 1000 entries in replay are ~500MB
    sac = SoftActorCritic(RobotEnvironment, replay_size=20000)
    print(sac.replay_buffer)
