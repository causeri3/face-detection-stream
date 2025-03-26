from utils.payloads import json_payload, image_payload

import numpy as np
import logging
import time
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from PIL import Image


def convert_to_model_format(frame: np.ndarray):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_model(repo_name="arnabdhar/YOLOv8-Face-Detection",
               filename="model.pt"):
    model_path = hf_hub_download(repo_id=repo_name, filename=filename)
    model = YOLO(model_path)
    return model


def predict(image,
            model,
            return_json=True,
            return_image=False):

    """Gives object detection for one image from yolo v7 via onnx runtime session
    :param image: Input image in bytes or as np array
    :param model: YOLO model
    :param return_json: returning json payload with results (tags, boxes, confidences)
    :param return_image: Returning image with labels, confidences and boxes as bytes
    :returns If none of the above return options is chosen the image will open locally"""

    start_time = time.time()
    pil_image = convert_to_model_format(image)
    output = model(pil_image)
    detected_objects = Detections.from_ultralytics(output[0])

    logging.debug("Detected {} objects".format(len(detected_objects)))

    if return_json and return_image:
        json_output = json_payload(detected_objects)
        bytes_output = image_payload(detected_objects, image)
        end_time = time.time()
        logging.debug("One detection event took {:.2f} seconds".format(end_time - start_time))
        return bytes_output, json_output

    if return_json:
        json_output = json_payload(detected_objects)
        end_time = time.time()
        logging.debug("One detection event took {:.2f} seconds".format(end_time - start_time))
        return json_output

    if return_image:
        bytes_output = image_payload(detected_objects, image)
        end_time = time.time()
        logging.debug("One detection event took {:.2f} seconds".format(end_time - start_time))
        return bytes_output

    else:
        output_bytes = image_payload(detected_objects, image)
        output_array = np.frombuffer(output_bytes, dtype=np.uint8)
        output_image = cv2.imdecode(output_array, flags=1)
        end_time = time.time()
        logging.info("One detection event took {:.2f} seconds".format(end_time - start_time))
        cv2.imshow('image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

