from utils.payloads import json_payload, image_payload
from utils.args import get_args
from utils.track import get_ids
from utils.target import EyesTarget


import numpy as np
import logging
import time
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from PIL import Image

args, unknown = get_args()

eyes_target = EyesTarget()

def convert_to_model_format(frame: np.ndarray):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_model(repo_name="arnabdhar/YOLOv8-Face-Detection",
               filename: str = "model.pt"):
    model_path = hf_hub_download(repo_id=repo_name, filename=filename)
    model = YOLO(model_path)
    return model


def predict(image: np.ndarray,
            model: YOLO,
            return_json: bool = True,
            return_image: bool = False):

    """Gives object detection for one image from yolo v7 via onnx runtime session
    :param image: Input image in bytes or as np array
    :param model: YOLO model
    :param return_json: returning json payload with results
    :param return_image: Returning image with labels, confidences and boxes as bytes
    :returns If none of the above return options is chosen the image will open locally"""

    start_time = time.time()
    pil_image = convert_to_model_format(image)
    output = model(pil_image,
                   conf=args.confidence_threshold,
                   iou=args.iou_threshold)
    detected_objects = Detections.from_ultralytics(output[0])

    logging.debug("Detected {} objects".format(len(detected_objects)))

    json_output = json_payload(detected_objects)
    track_ids = get_ids(json_output["bbs"], image)
    [d.update({'id': i}) for d, i in zip(json_output["objects"], track_ids)]
    x, y = eyes_target.update(json_output["objects"], image)
    json_output.update({'target_coordinates_xy': (x,y)})

    end_time = time.time()
    logging.debug("One detection event took {:.2f} seconds".format(end_time - start_time))
    if return_image:
        bytes_output = image_payload(detected_objects, image, track_ids, (x, y))
        return bytes_output, json_output

    return json_output



