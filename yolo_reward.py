# This file contains code to take in an image, run it through the YOLO ONNX network, and produce a "loss"
# We are shooting for an initial loss function to be "is there a detected object in the center of the frame"
from pathlib import Path
from typing import List

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import png


input_binding_name = "images"

# Get these from exploring your ONNX file, the first three are the three yolo-kernel result layers, the final entry
# is the intermediate layer to be passed into SAC
output_binding_names = ["output", "427", "446", "300"]

# YOLO Specific parameters
input_h = 480
input_w = 640
class_num = 80
check_count = 3

class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


@dataclass
class YoloKernel:
    width: int
    height: int
    anchors: tuple

@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int
    class_idx: int
    confidence: float

yolo1 = YoloKernel(input_w // 32, input_h // 32, (116,90,  156,198,  373,326))
yolo2 = YoloKernel(input_w // 16, input_h // 16, (30,61,  62,45,  59,119))
yolo3 = YoloKernel(input_w // 8, input_h // 8, (10,13,  16,30,  33,23))


def detect_yolo_bboxes(prediction: np.ndarray, kernel: YoloKernel) -> List[BBox]:
    result = []
    prediction = 1 / (1 + np.exp(-prediction))

    for c in range(check_count):
        for row in range(prediction.shape[2]):
            for col in range(prediction.shape[3]):
                box_prob = prediction[0, c, row, col, 4]

                if box_prob < 0.50:
                    continue

                for class_idx in range(class_num):
                    class_prob = prediction[0, c, row, col, 5 + class_idx]
                    class_prob *= box_prob

                    if class_prob > 0.80:
                        result.append(BBox(x=(col - 0.5 + 2 * prediction[0, c, row, col, 0]) * input_w / kernel.width,
                                           y=(row - 0.5 + 2 * prediction[0, c, row, col, 1]) * input_w / kernel.width,
                                           width=(2*prediction[0, c, row, col, 2])**2*kernel.anchors[2*c],
                                           height=(2*prediction[0, c, row, col, 3])**2*kernel.anchors[2*c + 1],
                                           class_idx=class_idx,
                                           confidence=class_prob))

    return result


def any_objects_present(prediction: np.ndarray, kernel: YoloKernel) -> float:
    prediction = 1 / (1 + np.exp(-prediction))

    all_probs = np.expand_dims(prediction[..., 4], -1) * prediction[..., 5:]

    return np.sum(all_probs) / 50.0


def centered_objects_present(prediction: np.ndarray, kernel: YoloKernel) -> float:
    prediction = 1 / (1 + np.exp(-prediction))
    all_probs = prediction[..., 4] * np.amax(prediction[..., 5:], axis=-1)

    xgrid = np.arange(prediction.shape[3]).reshape((1, 1, 1, -1))
    all_xs = (xgrid - 0.5 + 2 * prediction[..., 0]) * input_w / kernel.width

    ygrid = np.arange(prediction.shape[2]).reshape((1,1,-1,1))
    all_ys = (ygrid - 0.5 + 2 * prediction[..., 1]) * input_h / kernel.height

    all_centers = np.sqrt((all_xs - input_w / 2) ** 2 + (all_ys - input_h / 2) ** 2)

    return np.sum(all_probs / all_centers)


def get_reward(pred: List[np.ndarray]) -> float:
    # all_bboxes = detect_yolo_bboxes(pred[0], yolo1) + \
    #              detect_yolo_bboxes(pred[1], yolo2) + \
    #              detect_yolo_bboxes(pred[2], yolo3)
    #print(all_bboxes)

    return centered_objects_present(pred[0], yolo1) + \
           centered_objects_present(pred[1], yolo2) + \
           centered_objects_present(pred[2], yolo3)


def get_prediction(sess: rt.InferenceSession, png_path: Path) -> List[np.ndarray]:
    pngdata = png.Reader(filename=png_path).asRGB8()
    image_np = np.vstack(pngdata[2])

    # First shape it into the WxHx3 format
    image_np = image_np.reshape((1, pngdata[1], pngdata[0], 3))

    assert pngdata[0] == input_w
    assert pngdata[1] == input_h

    # Now convert to NCHW
    image_np = image_np.transpose((0, 3, 1, 2))
    # Scale to 0 to -1
    image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)

    print(image_np.shape)

    pred = sess.run(output_binding_names, {
        input_binding_name: image_np
    })

    return pred


def get_intermediate_layer(pred: List[np.ndarray]) -> np.ndarray:
    return pred[3]


if __name__ == "__main__":
    sess = rt.InferenceSession("C:\\Users\jakep\Desktop\yolov5s.onnx")
    pred = get_prediction(sess, Path("imgs", "1598832182944358229.png"))
    reward = get_reward(pred)
    print(f"Reward: {reward}")