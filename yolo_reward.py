# This file contains code to take in an image, run it through the YOLO ONNX network, and produce a "loss"
# We are shooting for an initial loss function to be "is there a detected object in the center of the frame"
from pathlib import Path
from typing import List

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import png


input_binding_name = "images"

# Get these from exploring your ONNX file, the first one should be the classifications, all concatenated together
# The second one should be your intermediate output layer
output_binding_names = ["output", "361"]

# YOLO Specific parameters
input_h = 480
input_w = 640
class_num = 80

class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]



@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float


def detect_yolo_bboxes(final_detections: np.ndarray) -> List[BBox]:
    # Expected shape is ([batch], N, 85), or a list of all possible detections to check in a given image
    # Format of the 85-tensor is [center x, center y, width, height,
    #                             box_probability, person_probability, bicycle_probability..., toothbrush_probability]
    # per the class names list above

    boxes = []

    if final_detections.shape[0] == 1:
        final_detections = np.squeeze(final_detections, axis=0)

    for pred in final_detections:
        box_prob = pred[4]

        if box_prob < 0.50:
            continue

        for class_idx in range(class_num):
            class_prob = pred[5 + class_idx]
            class_prob *= box_prob

            if class_prob > 0.50:
                boxes.append(BBox(x=pred[0] - pred[2] / 2,
                                  y=pred[1] - pred[3] / 2,
                                  width=pred[2],
                                  height=pred[3],
                                  class_name=class_names[class_idx],
                                  confidence=class_prob))

    return boxes


def any_objects_present(prediction: np.ndarray) -> float:
    prediction = 1 / (1 + np.exp(-prediction))

    all_probs = np.expand_dims(prediction[..., 4], -1) * prediction[..., 5:]

    return np.sum(all_probs) / 50.0


def centered_objects_present(prediction: np.ndarray) -> float:
    prediction = 1 / (1 + np.exp(-prediction))
    all_probs = prediction[..., 4] * np.amax(prediction[..., 5:], axis=-1)

    xgrid = np.arange(prediction.shape[3]).reshape((1, 1, 1, -1))
    all_xs = (xgrid - 0.5 + 2 * prediction[..., 0]) * input_w / kernel.width

    ygrid = np.arange(prediction.shape[2]).reshape((1,1,-1,1))
    all_ys = (ygrid - 0.5 + 2 * prediction[..., 1]) * input_h / kernel.height

    all_centers = np.sqrt((all_xs - input_w / 2) ** 2 + (all_ys - input_h / 2) ** 2)

    return np.sum(all_probs / all_centers)


def sum_centered_objects_present(pred: List[np.ndarray]) -> float:
    # all_bboxes = detect_yolo_bboxes(pred[0], yolo1) + \
    #              detect_yolo_bboxes(pred[1], yolo2) + \
    #              detect_yolo_bboxes(pred[2], yolo3)
    #print(all_bboxes)

    return centered_objects_present(pred[0], yolo1) + \
           centered_objects_present(pred[1], yolo2) + \
           centered_objects_present(pred[2], yolo3)

def sum_centered_objects_present_weighted_closer(pred: List[np.ndarray]) -> float:
    # all_bboxes = detect_yolo_bboxes(pred[0], yolo1) + \
    #              detect_yolo_bboxes(pred[1], yolo2) + \
    #              detect_yolo_bboxes(pred[2], yolo3)
    #print(all_bboxes)

    return centered_objects_present(pred[0], yolo1) * 0.1 + \
           centered_objects_present(pred[1], yolo2) * 0.5 + \
           centered_objects_present(pred[2], yolo3) * 0.9


def convert_wh_to_nchw(image_np: np.ndarray) -> np.ndarray:
    # First shape it into the 1xWxHx1 format
    # Go from (H, W) to (1, H, W, 1)
    image_np = np.expand_dims(image_np, 0)
    image_np = np.expand_dims(image_np, -1)

    # Now broadcast to 1xHxWx3
    image_np = image_np * np.ones(dtype=image_np.dtype, shape=(image_np.shape[0], image_np.shape[1], image_np.shape[2], 3))

    assert image_np.shape[2] == input_w
    assert image_np.shape[1] == input_h

    # Now convert to NCHW
    image_np = image_np.transpose((0, 3, 1, 2))
    # Scale to 0 to -1
    image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)

    return image_np


def get_onnx_prediction(sess: rt.InferenceSession, image_np: np.ndarray) -> List[np.ndarray]:
    image_np = convert_wh_to_nchw(image_np)

    pred = sess.run(output_binding_names, {
        input_binding_name: image_np
    })

    return pred


def get_intermediate_layer(pred: List[np.ndarray]) -> np.ndarray:
    return pred[1]

