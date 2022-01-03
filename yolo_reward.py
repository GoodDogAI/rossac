# This file contains code to take in an image, run it through the YOLO ONNX network, and produce a "loss"
# We are shooting for an initial loss function to be "is there a detected object in the center of the frame"
import os
import png
from typing import List

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import torchvision

import torch

input_binding_name = "images"

# Get these from exploring your ONNX file, the first one should be the classifications, all concatenated together
# The second one should be your intermediate output layer
output_binding_names = ["output", "532"]

# YOLO Specific parameters
input_h = 480
input_w = 640
class_num = 80

GLOBAL_REWARD_SCALE = 0.10

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


def load_png(png_path: str) -> np.ndarray:
    pngdata = png.Reader(filename=png_path).asRGBA8()
    image_np = np.vstack(pngdata[2])
    image_np = image_np[:, np.mod(np.arange(image_np.shape[1]), 4) != 3]  # Skip the alpha channels
    image_np = image_np.reshape((image_np.shape[0], image_np.shape[1] // 3, 3))
    return image_np


def detect_yolo_bboxes(final_detections: np.ndarray, threshold=0.50) -> List[BBox]:
    # Expected shape is ([batch], N, 85), or a list of all possible detections to check in a given image
    # Format of the 85-tensor is [center x, center y, width, height,
    #                             box_probability, person_probability, bicycle_probability..., toothbrush_probability]
    # per the class names list above

    boxes = []

    if final_detections.shape[0] == 1:
        final_detections = np.squeeze(final_detections, axis=0)

    for pred in final_detections:
        box_prob = pred[4]

        if box_prob < threshold:
            continue

        for class_idx in range(class_num):
            class_prob = pred[5 + class_idx]
            class_prob *= box_prob

            if class_prob >= threshold:
                boxes.append(BBox(x=pred[0] - pred[2] / 2,
                                  y=pred[1] - pred[3] / 2,
                                  width=pred[2],
                                  height=pred[3],
                                  class_name=class_names[class_idx],
                                  confidence=class_prob))

    return boxes


def non_max_supression(input_bboxes: np.ndarray, iou_threshold: float = 0.50) -> np.ndarray:
    assert input_bboxes.shape[0] == 1, "This operation cannot be batched"

    yolo_bboxes = torch.from_numpy(input_bboxes)[0]

    boxes = torchvision.ops.box_convert(yolo_bboxes[..., 0:4], in_fmt="cxcywh", out_fmt="xyxy")
    scores = yolo_bboxes[..., 4] * torch.amax(yolo_bboxes[..., 5:], dim=-1)

    # Remove boxes that are below some low threshold
    confidence_filter_mask = scores > 0.10
    boxes = boxes[confidence_filter_mask]
    scores = scores[confidence_filter_mask]
    original_indexes = torch.arange(0, yolo_bboxes.shape[0], dtype=torch.int64)[confidence_filter_mask]

    nms_boxes = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
    original_nms_boxes = original_indexes[nms_boxes]

    return input_bboxes[0, original_nms_boxes]


def _all_centers(bboxes: np.ndarray) -> np.ndarray:
    return np.sqrt(((bboxes[..., 0] - input_w / 2) / input_w) ** 2 +
                  ((bboxes[ ..., 1] - input_h / 2) / input_h) ** 2) + 0.1  # Small constant to prevent divide by zero explosion


def sum_centered_objects_present(bboxes: np.ndarray) -> float:
    all_probs = bboxes[..., 4] * np.amax(bboxes[..., 5:], axis=-1)
    all_centers = _all_centers(bboxes)

    return np.sum(all_probs / all_centers) * GLOBAL_REWARD_SCALE


def prioritize_centered_spoons_with_nms(bboxes: np.ndarray) -> float:
    bboxes = non_max_supression(bboxes)
    return prioritize_centered_objects(bboxes, class_weights={
        "person": 3,
        "spoon": 10,
    })


def prioritize_centered_objects(bboxes: np.ndarray, class_weights: dict) -> float:
    all_probs = bboxes[..., 4] * np.amax(bboxes[..., 5:], axis=-1)
    all_centers = _all_centers(bboxes)

    classes = np.argmax(bboxes[..., 5:], axis=-1)
    factors = np.ones_like(all_probs)

    for (cls, factor) in class_weights.items():
        factors *= np.where(classes == class_names.index(cls), factor, 1.0)

    return np.sum((all_probs * factors) / all_centers) * GLOBAL_REWARD_SCALE


def convert_hwc_to_nchw(image_np: np.ndarray) -> np.ndarray:
    # Go from (H, W, 3) to (1, H, W, 3)
    image_np = np.expand_dims(image_np, 0)

    assert image_np.shape[2] == input_w
    assert image_np.shape[1] == input_h

    # Now convert to NCHW
    image_np = image_np.transpose((0, 3, 1, 2))
    # Scale to 0 to -1
    image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)

    return image_np


def get_onnx_prediction(sess: rt.InferenceSession, image_np: np.ndarray) -> List[np.ndarray]:
    image_np = convert_hwc_to_nchw(image_np)

    pred = sess.run(output_binding_names, {
        input_binding_name: image_np
    })

    # Returns (detections, intermediate_layer)
    detections, intermediate = pred
    return [detections, intermediate]

