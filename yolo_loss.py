# This file contains code to take in an image, run it through the YOLO ONNX network, and produce a "loss"
# We are shooting for an initial loss function to be "is there a detected object in the center of the frame"
from pathlib import Path

import numpy as np
import onnxruntime as rt
import png

sess = rt.InferenceSession("C:\\Users\jakep\Desktop\yolov5s.onnx")


def get_loss(png_path: Path) -> float:
    pngdata = png.Reader(filename=png_path).asRGB8()
    image_np = np.vstack(pngdata[2])

    # First shape it into the WxHx3 format
    image_np = image_np.reshape((1, pngdata[0], pngdata[1], 3))
    # Now convert to NCHW
    image_np = image_np.transpose((0, 3, 2, 1))
    # Scale to 0 to -1
    image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)

    print(image_np.shape)

    pred = sess.run(None, {
        "images": image_np
    })

    print(len(pred))

if __name__ == "__main__":
    get_loss(Path("imgs", "1598832074653309807.png"))