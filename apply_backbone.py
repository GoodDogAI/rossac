import numpy as np
import png
import os, sys
import torch

# requires https://github.com/ArmyOfRobots/yolov5 to be cloned in ..\YOLOv5
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))

import models


def prepare_backbone(backbone):
    backbone.eval()
    backbone.model[-1].export = True

    # Don't fuse layers, it won't work with torchscript exports
    #model.fuse()

    for param in backbone.parameters():
        param.requires_grad = False
    
    # from https://stackoverflow.com/a/52548419/231238
    # cut last layer
    prepared = torch.nn.Sequential(*(list(backbone.model.children())[:-1]))
    ## assert(list(prepared.children())[-1].type == 'models.common.BottleneckCSP')

    def get_backbone_output(input):
        outputs = backbone(input)
        assert(len(outputs) == 4)
        return outputs[-1]

    return get_backbone_output

def apply_backbone(backbone, img_path, output_path):
    img_data = png.Reader(filename=img_path).asDirect()[2]
    data = np.vstack(map(np.uint8, img_data))
    torch_in = torch.from_numpy(data).reshape([480,640,3]).permute(2,0,1).reshape([1, 3, 480, 640]) / 255.0
    processed = backbone(torch_in)
    np.save(output_path, processed.numpy(), allow_pickle=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('img_dir')
    opt = parser.parse_args()

    backbone = torch.load(opt.model)['model'].float()
    backbone = prepare_backbone(backbone)

    import os

    for file_name in os.listdir(opt.img_dir):
        img_path = os.path.join(opt.img_dir, file_name)
        _, ext = os.path.splitext(file_name)
        if not(os.path.isfile(img_path)) or ext != '.png':
            continue

        apply_backbone(backbone, img_path, img_path + ".npy")